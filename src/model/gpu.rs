//! Full-GPU forward pass for LLaMA/Qwen2 models.
//!
//! All intermediate states and the KV cache live in VRAM. Only the final
//! logits are copied back to the host, so a token costs one synchronization
//! instead of ~97.

use std::sync::Arc;
use std::time::Instant;

use crate::cuda::{CudaBuffer, CudaDevice, QuantBuffer};
use crate::error::{Error, Result};
use crate::model::attention::rope_table;
use crate::model::config::{ModelConfig, RopeStyle};
use crate::model::weights::ModelWeights;
use crate::tensor::Matrix;

struct Layer {
    wq: QuantBuffer,
    wk: QuantBuffer,
    wv: QuantBuffer,
    wo: QuantBuffer,
    gate: QuantBuffer,
    up: QuantBuffer,
    down: QuantBuffer,
    attn_norm: CudaBuffer,
    bq: Option<CudaBuffer>,
    bk: Option<CudaBuffer>,
    bv: Option<CudaBuffer>,
    ffn_norm: CudaBuffer,
    kc: CudaBuffer,
    vc: CudaBuffer,
}

fn pad256(n: usize) -> usize { (n + 255) / 256 * 256 }

fn f32_to_half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7fffff;
    if exp <= 0 {
        if exp < -10 {
            sign
        } else {
            let m = (0x800000 | mant) >> (27 - exp);
            sign | (m as u16)
        }
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut e = exp;
        let mut m = (mant + 0x1000) >> 13;
        if m == 0x800 {
            m = 0;
            e += 1;
        }
        if e >= 31 {
            return sign | 0x7c00;
        }
        sign | ((e as u16) << 10) | (m as u16)
    }
}

fn upload_vec(dev: &CudaDevice, m: &Matrix) -> Result<CudaBuffer> {
    let mut v = vec![0f32; m.cols()];
    m.row(0, &mut v)?;
    dev.upload(&v)
}

fn upload_opt_vec(dev: &CudaDevice, m: Option<&Matrix>) -> Result<Option<CudaBuffer>> {
    match m {
        Some(m) => Ok(Some(upload_vec(dev, m)?)),
        None => Ok(None),
    }
}

fn upload_w(dev: &CudaDevice, m: &Matrix) -> Result<QuantBuffer> {
    // Dequantize once into f32. The coalesced `matvec_f32_rows` kernel is
    // much faster than per-block quantized dequantization for these shapes.
    let mut v = vec![0f32; m.rows() * m.cols()];
    for r in 0..m.rows() {
        m.row(r, &mut v[r * m.cols()..(r + 1) * m.cols()])?;
    }
    let mut half_bytes = vec![0u8; v.len() * 2];
    for (dst, src) in half_bytes.chunks_exact_mut(2).zip(&v) {
        dst.copy_from_slice(&f32_to_half(*src).to_le_bytes());
    }
    dev.upload_quant(1, m.rows(), m.cols(), &half_bytes)
}

/// GPU execution state, created once per model.
pub(crate) struct LlamaGpu {
    dev: Arc<CudaDevice>,
    layers: Vec<Layer>,
    output: QuantBuffer,
    final_norm: CudaBuffer,
    x: CudaBuffer,
    x2: CudaBuffer,
    xb: CudaBuffer,
    q: CudaBuffer,
    k: CudaBuffer,
    v: CudaBuffer,
    attn: CudaBuffer,
    o: CudaBuffer,
    gate: CudaBuffer,
    up: CudaBuffer,
    act: CudaBuffer,
    down: CudaBuffer,
    logits: CudaBuffer,
    cos: CudaBuffer,
    sin: CudaBuffer,
    pos_dev: CudaBuffer,
    calls: usize,
    n_embd: usize,
    n_ffn: usize,
    hd: usize,
    n_heads: usize,
    n_kv: usize,
    width: usize,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    half: usize,
}

impl LlamaGpu {
    pub fn new(
        dev: Arc<CudaDevice>,
        cfg: &ModelConfig,
        weights: &ModelWeights,
    ) -> Result<Self> {
        let n = cfg.n_embd;
        let hd = cfg.head_dim;
        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.kv_dim();
        let capacity = cfg.n_ctx.min(1024).max(64);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in &weights.layers {
            layers.push(Layer {
                wq: upload_w(&dev, &l.wq)?,
                wk: upload_w(&dev, &l.wk)?,
                wv: upload_w(&dev, &l.wv)?,
                wo: upload_w(&dev, &l.wo)?,
                gate: upload_w(&dev, &l.w_gate)?,
                up: upload_w(&dev, &l.w_up)?,
                down: upload_w(&dev, &l.w_down)?,
                attn_norm: upload_vec(&dev, &l.attn_norm)?,
                bq: upload_opt_vec(&dev, l.bq.as_ref())?,
                bk: upload_opt_vec(&dev, l.bk.as_ref())?,
                bv: upload_opt_vec(&dev, l.bv.as_ref())?,
                ffn_norm: upload_vec(&dev, &l.ffn_norm)?,
                kc: dev.alloc(capacity * kv_dim)?,
                vc: dev.alloc(capacity * kv_dim)?,
            });
        }
        let half = cfg.rope_dim / 2;
        let (cos_table, sin_table) = rope_table(
            cfg.n_ctx.min(8192),
            cfg.rope_dim,
            cfg.rope_base,
            RopeStyle::Llama,
            cfg.rope_scale,
        );
        // The 152k-row output head is dequantized once to f32: a coalesced
        // block-per-row kernel is much faster than scalar block dequantization
        // for this shape.
        let mut output_vals = vec![0f32; weights.output.rows() * weights.output.cols()];
        for r in 0..weights.output.rows() {
            weights.output.row(r, &mut output_vals[r * weights.output.cols()..(r + 1) * weights.output.cols()])?;
        }
        let mut output_half = vec![0u8; output_vals.len() * 2];
        for (dst, src) in output_half.chunks_exact_mut(2).zip(&output_vals) {
            dst.copy_from_slice(&f32_to_half(*src).to_le_bytes());
        }
        let output = dev.upload_quant(1, weights.output.rows(), weights.output.cols(), &output_half)?;
        let final_norm = upload_vec(&dev, &weights.final_norm)?;
        let x = dev.upload(&vec![0.0f32; pad256(n)])?;
        let x2 = dev.upload(&vec![0.0f32; pad256(n)])?;
        let xb = dev.upload(&vec![0.0f32; pad256(n)])?;
        let q = dev.upload(&vec![0.0f32; pad256(q_dim)])?;
        let k = dev.upload(&vec![0.0f32; pad256(kv_dim)])?;
        let v = dev.upload(&vec![0.0f32; pad256(kv_dim)])?;
        let attn = dev.upload(&vec![0.0f32; pad256(q_dim)])?;
        let o = dev.upload(&vec![0.0f32; pad256(n)])?;
        let gate = dev.upload(&vec![0.0f32; pad256(cfg.n_ffn)])?;
        let up = dev.upload(&vec![0.0f32; pad256(cfg.n_ffn)])?;
        let act = dev.upload(&vec![0.0f32; pad256(cfg.n_ffn)])?;
        let down = dev.upload(&vec![0.0f32; pad256(n)])?;
        let logits = dev.alloc(cfg.vocab_size)?;
        let cos = dev.alloc(half)?;
        let sin = dev.alloc(half)?;
        let pos_dev = dev.alloc(1)?;
        Ok(Self {
            dev,
            layers,
            output,
            final_norm,
            x,
            x2,
            xb,
            q,
            k,
            v,
            attn,
            o,
            gate,
            up,
            act,
            down,
            logits,
            cos,
            sin,
            pos_dev,
            calls: 0,
            n_embd: n,
            n_ffn: cfg.n_ffn,
            hd,
            n_heads: cfg.n_heads,
            n_kv: cfg.n_kv_heads,
            width: kv_dim,
            cos_table,
            sin_table,
            half,
        })
    }

    fn mv(&self, q: &QuantBuffer, x: &CudaBuffer, y: &CudaBuffer) -> Result<()> {
        if q.kind == 1 {
            self.dev.matvec_f16_rows_device(q, x, y)
        } else {
            self.dev.matvec_q_device(q, x, y)
        }
    }

    fn mv_out(&self, x: &CudaBuffer, y: &CudaBuffer, _rows: usize, _cols: usize) -> Result<()> {
        self.dev.matvec_f16_rows_device(&self.output, x, y)
    }

    pub fn forward(
        &mut self,
        weights: &ModelWeights,
        cfg: &ModelConfig,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        if token as usize >= cfg.vocab_size {
            return Err(Error::InvalidToken(token));
        }

        let t_start = Instant::now();
        self.calls += 1;
        let graph_mode = std::env::var_os("CORTEX_CUDA_GRAPH").is_some() && self.calls >= 2;
        let capture = graph_mode && !self.dev.graph_ready();

        // Embedding lookup stays on the host (one row, tiny).
        let mut emb = vec![0f32; self.n_embd];
        weights.token_embd.row(token as usize, &mut emb)?;
        // With graphs, keep input copies synchronous so no pending work can be
        // tangled with capture/replay on the same stream.
        if graph_mode {
            self.dev.copy_to_device(&self.x, &emb)?;
            self.dev.copy_to_device(&self.cos, &self.cos_table[pos * self.half..(pos + 1) * self.half])?;
            self.dev.copy_to_device(&self.sin, &self.sin_table[pos * self.half..(pos + 1) * self.half])?;
            self.dev.copy_bytes_to_device(&self.pos_dev, &(pos as i32).to_le_bytes())?;
        } else {
            self.dev.copy_to_device_async(&self.x, &emb)?;
            self.dev.copy_to_device_async(&self.cos, &self.cos_table[pos * self.half..(pos + 1) * self.half])?;
            self.dev.copy_to_device_async(&self.sin, &self.sin_table[pos * self.half..(pos + 1) * self.half])?;
            self.dev.copy_bytes_to_device_async(&self.pos_dev, &(pos as i32).to_le_bytes())?;
        }
        if capture {
            self.dev.activate_graph_stream();
            self.dev.graph_begin()?;
        }

        let mut cur = &self.x;
        let mut next = &self.x2;
        for (li, layer) in self.layers.iter().enumerate() {
            self.dev.rmsnorm_device(cur, &layer.attn_norm, &self.xb, self.n_embd, cfg.rms_eps)?;
            self.dev.matvec_qkv_device(
                &layer.wq, &layer.wk, &layer.wv,
                &self.xb, &self.q, &self.k, &self.v,
            )?;
            if let Some(b) = &layer.bq {
                self.dev.add_bias_device(&self.q, b, self.n_heads * self.hd)?;
            }
            if let Some(b) = &layer.bk {
                self.dev.add_bias_device(&self.k, b, self.width)?;
            }
            if let Some(b) = &layer.bv {
                self.dev.add_bias_device(&self.v, b, self.width)?;
            }
            self.dev.rope_device(
                &self.q, &self.k, &self.cos, &self.sin,
                self.n_heads, self.n_kv, self.hd,
            )?;
            self.dev.append_kv_device(
                &layer.kc, &layer.vc, &self.k, &self.v, &self.pos_dev, self.width,
            )?;
            self.dev.attention_device(
                &self.q, &layer.kc, &layer.vc, &self.attn,
                &self.pos_dev, self.n_heads, self.n_kv, self.hd, self.width,
                0,
            )?;
            self.mv(&layer.wo, &self.attn, &self.o)?;
            self.dev.add_device(cur, &self.o, next, self.n_embd)?;
            std::mem::swap(&mut cur, &mut next);

            self.dev.rmsnorm_device(cur, &layer.ffn_norm, &self.xb, self.n_embd, cfg.rms_eps)?;
            self.dev.matvec_gu_device(
                &layer.gate, &layer.up, &self.xb, &self.gate, &self.up,
            )?;
            self.dev.swiglu_device(&self.gate, &self.up, &self.act, self.n_ffn)?;
            self.mv(&layer.down, &self.act, &self.down)?;
            self.dev.add_device(cur, &self.down, next, self.n_embd)?;
            std::mem::swap(&mut cur, &mut next);
            let _ = li;
        }

        let t_layer = t_start.elapsed();
        self.dev.rmsnorm_device(cur, &self.final_norm, &self.xb, self.n_embd, cfg.rms_eps)?;
        self.mv_out(&self.xb, &self.logits, cfg.vocab_size, self.n_embd)?;
        if capture {
            self.dev.graph_end()?;
            self.dev.activate_main_stream();
            self.dev.graph_launch()?;
            self.dev.stream_sync()?;
        } else if graph_mode && self.dev.graph_ready() {
            self.dev.graph_launch()?;
            self.dev.stream_sync()?;
        } else {
            self.dev.synchronize()?;
        }
        let t_all = t_start.elapsed();
        if std::env::var_os("CORTEX_GPU_PROFILE").is_some() {
            eprintln!("[gpu-profile] layers={:?} total={:?}", t_layer, t_all);
        }
        let mut logits = vec![0f32; cfg.vocab_size];
        self.dev.copy_to_host(&mut logits, &self.logits)?;
        Ok(logits)
    }
}
