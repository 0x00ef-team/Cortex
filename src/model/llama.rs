//! LLaMA-style transformer inference.

use std::path::Path;

use crate::cuda::{CudaBuffer, CudaDevice};
use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::Device;
use std::sync::Arc;

use crate::tensor::{dot, DequantPolicy};
use crate::thread_pool::ThreadPool;

use super::attention::{rmsnorm, rope_table, rotate_half, softmax, swiglu};
use super::cache::LayerCache;
use super::config::{ModelConfig, RopeStyle};
use super::weights::ModelWeights;

/// Options controlling model loading.
pub struct LoadOptions {
    /// Keep weights quantized (default) or expand them to `f32` at load time.
    pub dequant: DequantPolicy,
    /// Override the context length stored in the file (useful to save RAM).
    pub context: Option<usize>,
    /// Override the rotary embedding convention.
    pub rope_style: Option<RopeStyle>,
    /// Number of threads used for large matrix-vector products (0 = auto).
    pub threads: usize,
    /// Compute device. CUDA mode forces eager dequantization.
    pub device: Device,
    /// Called with human-readable progress messages.
    pub progress: Option<Box<dyn FnMut(&str)>>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dequant: DequantPolicy::Lazy,
            context: None,
            rope_style: None,
            threads: 0,
            device: Device::Cpu,
            progress: None,
        }
    }
}

/// Weight matrices uploaded to VRAM.
struct GpuLayer {
    wq: CudaBuffer,
    wk: CudaBuffer,
    wv: CudaBuffer,
    wo: CudaBuffer,
    w_gate: CudaBuffer,
    w_up: CudaBuffer,
    w_down: CudaBuffer,
}

struct GpuWeights {
    layers: Vec<GpuLayer>,
    output: CudaBuffer,
}

/// A loaded model ready for autoregressive decoding.
pub struct Llama {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    cache: Vec<LayerCache>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pool: Arc<ThreadPool>,
    gpu: Option<Arc<CudaDevice>>,
    gpu_weights: Option<GpuWeights>,
}

impl Llama {
    /// Load a GGUF model.
    pub fn load(path: impl AsRef<Path>, mut options: LoadOptions) -> Result<Self> {
        let mut gguf = GgufFile::load(path)?;

        if let Some(p) = &mut options.progress {
            p("reading GGUF metadata");
        }
        let arch = gguf
            .get_str("general.architecture")
            .ok_or_else(|| Error::InvalidConfig("missing general.architecture".into()))?;
        let vocab_size = gguf
            .get_u32(&format!("{arch}.vocab_size"))
            .or_else(|| {
                gguf.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() as u32)
            })
            .ok_or_else(|| Error::InvalidConfig("cannot determine vocabulary size".into()))?
            as usize;

        let mut config = ModelConfig::from_gguf(&gguf, vocab_size)?;
        if let Some(n) = options.context {
            config.n_ctx = config.n_ctx.min(n);
        }
        if let Some(style) = options.rope_style {
            config.rope_style = style;
        }

        if let Some(p) = &mut options.progress {
            p(&format!(
                "{}: {} layers, {} heads ({} KV), dim {}, ctx {}",
                config.architecture,
                config.n_layers,
                config.n_heads,
                config.n_kv_heads,
                config.n_embd,
                config.n_ctx
            ));
        }
        let policy = if options.device == Device::Cpu {
            options.dequant
        } else {
            DequantPolicy::Eager
        };
        let weights = ModelWeights::load(&mut gguf, &config, policy)?;
        if let Some(p) = &mut options.progress {
            p("weights loaded");
        }

        let (cos, sin) = rope_table(
            config.n_ctx,
            config.rope_dim,
            config.rope_base,
            config.rope_style,
            config.rope_scale,
        );
        let cache = (0..config.n_layers)
            .map(|_| LayerCache::new(64, config.kv_dim()))
            .collect();

        let threads = if options.threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            options.threads.max(1)
        };

        let (gpu, gpu_weights) = match options.device {
            Device::Cpu => (None, None),
            Device::Cuda(idx) => {
                if let Some(p) = &mut options.progress {
                    p("initializing CUDA and compiling kernels");
                }
                let device = Arc::new(CudaDevice::new()?);
                let _ = idx; // single-device support for now
                let upload = |m: &crate::tensor::Matrix| -> Result<CudaBuffer> {
                    let data = m
                        .as_f32()
                        .ok_or_else(|| Error::InvalidConfig("CUDA requires Eager weights".into()))?;
                    device.upload(data)
                };
                let mut layers = Vec::with_capacity(config.n_layers);
                for lw in &weights.layers {
                    layers.push(GpuLayer {
                        wq: upload(&lw.wq)?,
                        wk: upload(&lw.wk)?,
                        wv: upload(&lw.wv)?,
                        wo: upload(&lw.wo)?,
                        w_gate: upload(&lw.w_gate)?,
                        w_up: upload(&lw.w_up)?,
                        w_down: upload(&lw.w_down)?,
                    });
                }
                let gw = GpuWeights {
                    layers,
                    output: upload(&weights.output)?,
                };
                if let Some(p) = &mut options.progress {
                    p("weights uploaded to VRAM");
                }
                (Some(device), Some(gw))
            }
        };

        Ok(Self {
            config,
            weights,
            cache,
            cos,
            sin,
            pool: Arc::new(ThreadPool::new(threads)),
            gpu,
            gpu_weights,
        })
    }

    /// Number of tokens the KV cache can hold.
    pub fn context_length(&self) -> usize {
        self.config.n_ctx
    }

    /// Number of threads used for matrix-vector products.
    pub fn threads(&self) -> usize {
        self.pool.len()
    }

    /// CUDA `(total ns, calls)` if the GPU backend is active.
    pub fn cuda_stats(&self) -> Option<(u64, u64)> {
        self.gpu.as_ref().map(|d| d.stats())
    }

    /// Per-shape CUDA matvec statistics.
    pub fn cuda_stats_by_shape(&self) -> Option<std::collections::HashMap<(usize, usize), (u64, u64)>> {
        self.gpu.as_ref().map(|d| d.stats_by_shape())
    }

    /// Drop all cached keys/values so the model can be reused for a new prompt.
    pub fn reset_cache(&mut self) {
        for c in &mut self.cache {
            c.clear();
        }
    }

    /// Run the transformer for one token at position `pos` and return logits.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        if token as usize >= cfg.vocab_size {
            return Err(Error::InvalidToken(token));
        }
        if pos >= cfg.n_ctx {
            return Err(Error::ContextOverflow {
                pos,
                capacity: cfg.n_ctx,
            });
        }

        let hd = cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let n_heads = cfg.n_heads;
        let n_kv = cfg.n_kv_heads;
        let kv_mul = n_heads / n_kv;
        let half = cfg.rope_dim / 2;

        let mut x = vec![0.0f32; cfg.n_embd];
        self.weights.token_embd.row(token as usize, &mut x)?;

        let mut xb = vec![0.0f32; cfg.n_embd];
        let mut q = vec![0.0f32; n_heads * hd];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn = vec![0.0f32; cfg.n_embd];
        let mut o = vec![0.0f32; cfg.n_embd];
        let mut gate = vec![0.0f32; cfg.n_ffn];
        let mut up = vec![0.0f32; cfg.n_ffn];
        let mut act = vec![0.0f32; cfg.n_ffn];
        let mut down = vec![0.0f32; cfg.n_embd];
        let mut norm_buf = vec![0.0f32; cfg.n_embd];
        let mut scores = vec![0.0f32; pos + 1];

        let cos = &self.cos[pos * half..(pos + 1) * half];
        let sin = &self.sin[pos * half..(pos + 1) * half];

        // Disjoint borrows: immutable weights, mutable KV cache.
        let weights = &self.weights;
        let cache = &mut self.cache;
        let pool = Arc::clone(&self.pool);
        let gpu = self.gpu.as_ref();
        let gw = self.gpu_weights.as_ref();
        let scale = 1.0 / (hd as f32).sqrt();
        let mv = |buf: Option<&CudaBuffer>,
                  m: &crate::tensor::Matrix,
                  x: &[f32],
                  out: &mut [f32]|
         -> Result<()> {
            match (gpu, buf) {
                (Some(d), Some(b)) => d.matvec(b, x, out),
                _ => m.matvec_pool(x, out, &pool),
            }
        };
        let mvs = |bufs: &[Option<&CudaBuffer>],
                   mats: &[&crate::tensor::Matrix],
                   x: &[f32],
                   outs: &mut [&mut [f32]]|
         -> Result<()> {
            if let Some(d) = gpu {
                let refs: Vec<&CudaBuffer> = bufs.iter().flatten().copied().collect();
                d.matvec_many(&refs, x, outs)
            } else {
                for (m, o) in mats.iter().zip(outs.iter_mut()) {
                    m.matvec_pool(x, o, &pool)?;
                }
                Ok(())
            }
        };

        for li in 0..cfg.n_layers {
            let lw = &weights.layers[li];

            // --- Multi-head attention with grouped-query support ---
            lw.attn_norm.row(0, &mut norm_buf)?;
            rmsnorm(&x, &norm_buf, cfg.rms_eps, &mut xb)?;
            let gl = gw.map(|g| &g.layers[li]);
            mvs(
                &[gl.map(|g| &g.wq), gl.map(|g| &g.wk), gl.map(|g| &g.wv)],
                &[&lw.wq, &lw.wk, &lw.wv],
                &xb,
                &mut [&mut q, &mut k, &mut v],
            )?;
            if let Some(b) = &lw.bq {
                let mut bias = vec![0.0f32; q.len()];
                b.row(0, &mut bias)?;
                for i in 0..q.len() {
                    q[i] += bias[i];
                }
            }
            if let Some(b) = &lw.bk {
                let mut bias = vec![0.0f32; k.len()];
                b.row(0, &mut bias)?;
                for i in 0..k.len() {
                    k[i] += bias[i];
                }
            }
            if let Some(b) = &lw.bv {
                let mut bias = vec![0.0f32; v.len()];
                b.row(0, &mut bias)?;
                for i in 0..v.len() {
                    v[i] += bias[i];
                }
            }

            for h in 0..n_heads {
                rotate_half(&mut q, h * hd, hd, cos, sin, cfg.rope_style);
            }
            for h in 0..n_kv {
                rotate_half(&mut k, h * hd, hd, cos, sin, cfg.rope_style);
            }
            cache[li].append(pos, &k, &v);

            for h in 0..n_heads {
                let kv_head = h / kv_mul;
                let qh = &q[h * hd..(h + 1) * hd];
                for t in 0..=pos {
                    scores[t] = dot(qh, cache[li].key(t, kv_head, hd)) * scale;
                }
                softmax(&mut scores);
                let ah = &mut attn[h * hd..(h + 1) * hd];
                ah.fill(0.0);
                for t in 0..=pos {
                    let a = scores[t];
                    let vh = cache[li].value(t, kv_head, hd);
                    for d in 0..hd {
                        ah[d] += a * vh[d];
                    }
                }
            }

            mv(gl.map(|g| &g.wo), &lw.wo, &attn, &mut o)?;
            for i in 0..cfg.n_embd {
                x[i] += o[i];
            }

            // --- SwiGLU feed-forward network ---
            lw.ffn_norm.row(0, &mut norm_buf)?;
            rmsnorm(&x, &norm_buf, cfg.rms_eps, &mut xb)?;
            if let (Some(d), Some(gl)) = (gpu, gl) {
                d.ffn(&gl.w_gate, &gl.w_up, &gl.w_down, &xb, &mut down, cfg.n_ffn)?;
            } else {
                mvs(
                    &[],
                    &[&lw.w_gate, &lw.w_up],
                    &xb,
                    &mut [&mut gate, &mut up],
                )?;
                swiglu(&gate, &up, &mut act)?;
                mv(None, &lw.w_down, &act, &mut down)?;
            }
            for i in 0..cfg.n_embd {
                x[i] += down[i];
            }
        }

        weights.final_norm.row(0, &mut norm_buf)?;
        rmsnorm(&x, &norm_buf, cfg.rms_eps, &mut xb)?;

        let mut logits = vec![0.0f32; cfg.vocab_size];
        if let (Some(d), Some(g)) = (gpu, gw) {
            d.matvec_tall(&g.output, &xb, &mut logits)?;
        } else {
            mv(None, &weights.output, &xb, &mut logits)?;
        }
        Ok(logits)
    }
}
