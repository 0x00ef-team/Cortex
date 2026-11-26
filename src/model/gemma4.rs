//! Gemma 4 inference (dense text-only GGUFs).
//!
//! Gemma 4 differs substantially from LLaMA: alternating sliding-window and
//! global layers with different head dimensions, per-head Q/K RMSNorm,
//! per-layer token embeddings (PLE), post-attention/post-FFN norms, layer
//! output scales, shared KV layers and final-logit softcapping.

use std::path::Path;

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::model::attention::{rmsnorm, rotate_half, softmax};
use crate::model::config::RopeStyle;
use crate::model::cache::LayerCache;
use crate::tensor::{dot, DequantPolicy, Matrix};

const N_HEAD: usize = 8;
const N_KV: usize = 2;

#[derive(Clone, Debug)]
pub struct Gemma4Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub n_ffn: usize,
    pub n_embd_per_layer: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub softcap: f32,
    pub is_swa: Vec<bool>,
    pub head_dim_swa: usize,
    pub head_dim_global: usize,
    pub rope_base_swa: f32,
    pub rope_base_global: f32,
    pub shared_kv_from: usize,
    pub window: usize,
}

impl Gemma4Config {
    fn from_gguf(g: &GgufFile, vocab: usize) -> Result<Self> {
        let p = |s: &str| format!("gemma4.{s}");
        let get = |s: &str| g.get_u32(&p(s));
        let n_layers = get("block_count")
            .ok_or_else(|| Error::InvalidConfig("missing gemma4.block_count".into()))? as usize;
        let n_embd = get("embedding_length")
            .ok_or_else(|| Error::InvalidConfig("missing embedding_length".into()))? as usize;
        let n_ffn = get("feed_forward_length").unwrap_or((n_embd * 4) as u32) as usize;
        let n_embd_per_layer =
            get("embedding_length_per_layer_input").unwrap_or(0) as usize;
        let n_ctx = get("context_length").unwrap_or(4096) as usize;
        let eps = g.get_f32(&p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let softcap = g.get_f32(&p("final_logit_softcapping")).unwrap_or(0.0);
        let head_dim_swa = get("attention.key_length_swa").unwrap_or(256) as usize;
        let head_dim_global = get("attention.key_length").unwrap_or(512) as usize;
        let rope_base_swa = g.get_f32(&p("rope.freq_base_swa")).unwrap_or(10_000.0);
        let rope_base_global = g.get_f32(&p("rope.freq_base")).unwrap_or(1_000_000.0);
        let shared = get("attention.shared_kv_layers").unwrap_or(0) as usize;
        let window = get("attention.sliding_window").unwrap_or(512) as usize;

        let is_swa: Vec<bool> = g
            .metadata
            .get(&p("attention.sliding_window_pattern"))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v.as_bool().unwrap_or(true)).collect())
            .unwrap_or_else(|| vec![true; n_layers]);
        if is_swa.len() != n_layers {
            return Err(Error::InvalidConfig("sliding_window_pattern length mismatch".into()));
        }
        if n_embd_per_layer == 0 {
            return Err(Error::InvalidConfig("Gemma 4 requires per-layer embeddings".into()));
        }
        Ok(Self {
            n_layers,
            n_embd,
            vocab,
            n_ffn,
            n_embd_per_layer,
            n_ctx,
            eps,
            softcap,
            is_swa,
            head_dim_swa,
            head_dim_global,
            rope_base_swa,
            rope_base_global,
            shared_kv_from: n_layers.saturating_sub(shared),
            window,
        })
    }

    fn head_dim(&self, il: usize) -> usize {
        if self.is_swa[il] {
            self.head_dim_swa
        } else {
            self.head_dim_global
        }
    }

    fn has_kv(&self, il: usize) -> bool {
        il < self.shared_kv_from
    }

    fn kv_source(&self, il: usize) -> usize {
        if self.has_kv(il) {
            il
        } else {
            self.shared_kv_from - if self.is_swa[il] { 2 } else { 1 }
        }
    }
}

struct LayerWeights {
    attn_norm: Matrix,
    wq: Matrix,
    wk: Option<Matrix>,
    wv: Option<Matrix>,
    wo: Matrix,
    q_norm: Matrix,
    k_norm: Option<Matrix>,
    attn_post_norm: Matrix,
    ffn_norm: Matrix,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
    ffn_post_norm: Matrix,
    inp_gate: Matrix,
    proj: Matrix,
    post_norm: Matrix,
    out_scale: f32,
}

struct Weights {
    token_embd: Matrix,
    per_layer_token_embd: Matrix,
    per_layer_model_proj: Matrix,
    per_layer_proj_norm: Matrix,
    output_norm: Matrix,
    output: Matrix,
    rope_freqs: Option<Vec<f32>>,
    layers: Vec<LayerWeights>,
}

fn load_vec(g: &GgufFile, name: &str) -> Result<Matrix> {
    let cols = g.info(name)?.cols();
    let kind = g.info(name)?.kind;
    let (map, offset, len) = g.tensor_view(name)?;
    Matrix::from_gguf_mapped(kind, 1, cols, map, offset, len, DequantPolicy::Lazy)
}

fn load_mat(g: &GgufFile, name: &str) -> Result<Matrix> {
    let rows = g.info(name)?.rows();
    let cols = g.info(name)?.cols();
    let kind = g.info(name)?.kind;
    let (map, offset, len) = g.tensor_view(name)?;
    Matrix::from_gguf_mapped(kind, rows, cols, map, offset, len, DequantPolicy::Lazy)
}

fn load_opt(g: &mut GgufFile, name: &str) -> Result<Option<Matrix>> {
    if g.info(name).is_err() {
        return Ok(None);
    }
    Ok(Some(load_mat(g, name)?))
}

impl Weights {
    fn load(g: &mut GgufFile, cfg: &Gemma4Config) -> Result<Self> {
        let token_embd = load_mat(g, "token_embd.weight")?;
        let per_layer_token_embd = load_mat(g, "per_layer_token_embd.weight")?;
        let per_layer_model_proj = load_mat(g, "per_layer_model_proj.weight")?;
        let per_layer_proj_norm = load_vec(g, "per_layer_proj_norm.weight")?;
        let output_norm = load_vec(g, "output_norm.weight")?;
        let output = match g.info("output.weight") {
            Ok(_) => load_mat(g, "output.weight")?,
            Err(_) => load_mat(g, "token_embd.weight")?,
        };
        let rope_freqs = match g.info("rope_freqs.weight") {
            Ok(info) => {
                let cols = info.cols();
                let kind = info.kind;
                let mut v = vec![0f32; cols];
                let bytes = g.tensor_bytes("rope_freqs.weight")?;
                let m = Matrix::from_gguf(kind, 1, cols, &bytes, DequantPolicy::Lazy)?;
                m.row(0, &mut v)?;
                Some(v)
            }
            Err(_) => None,
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let q_dim = N_HEAD * cfg.head_dim(il);
            let kv_dim = N_KV * cfg.head_dim(il);
            let p = |s: &str| format!("blk.{il}.{s}");
            let scale = match load_opt(g, &p("layer_output_scale.weight"))? {
                Some(m) => {
                    let mut v = [0f32; 1];
                    m.row(0, &mut v)?;
                    v[0]
                }
                None => 1.0,
            };
            layers.push(LayerWeights {
                attn_norm: load_mat(g, &p("attn_norm.weight"))?,
                wq: load_mat(g, &p("attn_q.weight"))?,
                wk: load_opt(g, &p("attn_k.weight"))?,
                wv: load_opt(g, &p("attn_v.weight"))?,
                wo: load_mat(g, &p("attn_output.weight"))?,
                q_norm: load_vec(g, &p("attn_q_norm.weight"))?,
                k_norm: load_opt(g, &p("attn_k_norm.weight"))?,
                attn_post_norm: load_mat(g, &p("post_attention_norm.weight"))?,
                ffn_norm: load_mat(g, &p("ffn_norm.weight"))?,
                w_gate: load_mat(g, &p("ffn_gate.weight"))?,
                w_up: load_mat(g, &p("ffn_up.weight"))?,
                w_down: load_mat(g, &p("ffn_down.weight"))?,
                ffn_post_norm: load_mat(g, &p("post_ffw_norm.weight"))?,
                inp_gate: load_mat(g, &p("inp_gate.weight"))?,
                proj: load_mat(g, &p("proj.weight"))?,
                post_norm: load_mat(g, &p("post_norm.weight"))?,
                out_scale: scale,
            });
            let _ = (q_dim, kv_dim);
        }
        Ok(Self {
            token_embd,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            output_norm,
            output,
            rope_freqs,
            layers,
        })
    }
}

/// Gemma 4 model ready for autoregressive decoding (CPU).
pub struct Gemma4 {
    pub config: Gemma4Config,
    weights: Weights,
    cache: Vec<LayerCache>,
    cos_swa: Vec<f32>,
    sin_swa: Vec<f32>,
    cos_g: Vec<f32>,
    sin_g: Vec<f32>,
}

impl Gemma4 {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut g = GgufFile::load(path)?;
        let vocab = g
            .get_u32("gemma4.vocab_size")
            .or_else(|| {
                g.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() as u32)
            })
            .ok_or_else(|| Error::InvalidConfig("cannot determine vocab".into()))?
            as usize;
        let config = Gemma4Config::from_gguf(&g, vocab)?;
        let weights = Weights::load(&mut g, &config)?;

        let capacity = config.n_ctx.min(1024).max(64);
        let cache = (0..config.n_layers)
            .map(|il| LayerCache::new(capacity, N_KV * config.head_dim(il)))
            .collect();

        let (cos_swa, sin_swa) = rope_table(config.n_ctx.min(8192), config.head_dim_swa, config.rope_base_swa, None);
        let (cos_g, sin_g) = match &weights.rope_freqs {
            Some(freqs) => learned_rope_table(
                config.n_ctx.min(8192),
                config.head_dim_global,
                config.rope_base_global,
                freqs,
            ),
            None => rope_table(config.n_ctx.min(8192), config.head_dim_global, config.rope_base_global, None),
        };

        Ok(Self {
            config,
            weights,
            cache,
            cos_swa,
            sin_swa,
            cos_g,
            sin_g,
        })
    }

    pub fn reset_cache(&mut self) {
        for c in &mut self.cache {
            c.clear();
        }
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        if token as usize >= cfg.vocab || pos >= cfg.n_ctx {
            return Err(Error::InvalidConfig("token/position out of range".into()));
        }
        let n = cfg.n_embd;
        let scale_emb = (n as f32).sqrt();
        let mut x = vec![0f32; n];
        self.weights.token_embd.row(token as usize, &mut x)?;
        for v in &mut x {
            *v *= scale_emb;
        }

        // Per-layer inputs: PLE = (normed projection + token PLE) * sqrt(d/2)
        let ple_dim = cfg.n_embd_per_layer * cfg.n_layers;
        let mut ple_token = vec![0f32; ple_dim];
        self.weights.per_layer_token_embd.row(token as usize, &mut ple_token)?;
        let mut ple_proj = vec![0f32; ple_dim];
        self.weights.per_layer_model_proj.matvec(&x, &mut ple_proj)?;
        let proj_eps = cfg.eps * n as f32;
        let mut pbuf = vec![0f32; cfg.n_embd_per_layer];
        let mut pnorm = vec![0f32; cfg.n_embd_per_layer];
        self.weights.per_layer_proj_norm.row(0, &mut pnorm)?;
        for v in &mut pnorm {
            *v *= 1.0 / 2.0_f32.sqrt();
        }
        for il in 0..cfg.n_layers {
            let off = il * cfg.n_embd_per_layer;
            pbuf.copy_from_slice(&ple_proj[off..off + cfg.n_embd_per_layer]);
            // rmsnorm with weight, then add token PLE
            let mut normed = vec![0f32; cfg.n_embd_per_layer];
            rmsnorm(&pbuf, &pnorm, proj_eps, &mut normed)?;
            for j in 0..cfg.n_embd_per_layer {
                ple_token[off + j] = (normed[j] + ple_token[off + j])
                    * (cfg.n_embd_per_layer as f32 / 2.0).sqrt();
            }
        }

        let weights = &self.weights;
        let cache = &mut self.cache;
        let mut nbuf = vec![0f32; n];
        let mut normed = vec![0f32; n];
        let mut q = vec![0f32; 0];
        let mut k = vec![0f32; 0];
        let mut v = vec![0f32; 0];
        let mut attn = vec![0f32; 0];
        let mut o = vec![0f32; n];
        let mut gate = vec![0f32; cfg.n_ffn];
        let mut up = vec![0f32; cfg.n_ffn];
        let mut act = vec![0f32; cfg.n_ffn];
        let mut ffn = vec![0f32; n];
        let mut ple_gate = vec![0f32; cfg.n_embd_per_layer];
        let mut ple_out = vec![0f32; n];
        let mut ple_normed = vec![0f32; n];
        let mut scores = vec![0f32; pos + 1];

        for il in 0..cfg.n_layers {
            let lw = &weights.layers[il];
            let hd = cfg.head_dim(il);
            let q_dim = N_HEAD * hd;
            let kv_dim = N_KV * hd;
            let half = hd / 2;
            let (cos, sin) = if cfg.is_swa[il] {
                (&self.cos_swa[pos * half..(pos + 1) * half], &self.sin_swa[pos * half..(pos + 1) * half])
            } else {
                (&self.cos_g[pos * half..(pos + 1) * half], &self.sin_g[pos * half..(pos + 1) * half])
            };

            lw.attn_norm.row(0, &mut nbuf)?;
            rmsnorm(&x, &nbuf, cfg.eps, &mut normed)?;
            q.resize(q_dim, 0.0);
            lw.wq.matvec(&normed, &mut q)?;
            // per-head Q norm
            let mut qnorm_w = vec![0f32; hd];
            lw.q_norm.row(0, &mut qnorm_w)?;
            let mut qh = vec![0f32; hd];
            for h in 0..N_HEAD {
                qh.copy_from_slice(&q[h * hd..(h + 1) * hd]);
                let mut out = vec![0f32; hd];
                rmsnorm(&qh, &qnorm_w, cfg.eps, &mut out)?;
                q[h * hd..(h + 1) * hd].copy_from_slice(&out);
                rotate_half(&mut q, h * hd, hd, cos, sin, RopeStyle::NeoX);
            }

            let src = cfg.kv_source(il);
            if cfg.has_kv(il) {
                k.resize(kv_dim, 0.0);
                v.resize(kv_dim, 0.0);
                lw.wk
                    .as_ref()
                    .ok_or_else(|| Error::MissingTensor(format!("blk.{il}.attn_k.weight")))?
                    .matvec(&normed, &mut k)?;
                match &lw.wv {
                    Some(w) => w.matvec(&normed, &mut v)?,
                    None => v.copy_from_slice(&k),
                }
                if let Some(kn) = &lw.k_norm {
                    let mut knw = vec![0f32; hd];
                    kn.row(0, &mut knw)?;
                    let mut kh = vec![0f32; hd];
                    let one = vec![1.0f32; hd];
                    for h in 0..N_KV {
                        kh.copy_from_slice(&k[h * hd..(h + 1) * hd]);
                        let mut out = vec![0f32; hd];
                        rmsnorm(&kh, &knw, cfg.eps, &mut out)?;
                        k[h * hd..(h + 1) * hd].copy_from_slice(&out);
                        kh.copy_from_slice(&v[h * hd..(h + 1) * hd]);
                        rmsnorm(&kh, &one, cfg.eps, &mut out)?;
                        v[h * hd..(h + 1) * hd].copy_from_slice(&out);
                        rotate_half(&mut k, h * hd, hd, cos, sin, RopeStyle::NeoX);
                    }
                }
                cache[il].append(pos, &k, &v);
            }
            let kv_layer = &cache[src];
            let t_start = if cfg.is_swa[il] {
                pos.saturating_sub(cfg.window.saturating_sub(1))
            } else {
                0
            };
            attn.resize(q_dim, 0.0);
            for h in 0..N_HEAD {
                let kv_head = h / (N_HEAD / N_KV);
                let qh = &q[h * hd..(h + 1) * hd];
                let count = pos + 1 - t_start;
                scores[..count].fill(0.0);
                for (i, t) in (t_start..=pos).enumerate() {
                    scores[i] = dot(qh, kv_layer.key(t, kv_head, hd));
                }
                softmax(&mut scores[..count]);
                let ah = &mut attn[h * hd..(h + 1) * hd];
                ah.fill(0.0);
                for (i, t) in (t_start..=pos).enumerate() {
                    let a = scores[i];
                    let vh = kv_layer.value(t, kv_head, hd);
                    for j in 0..hd {
                        ah[j] += a * vh[j];
                    }
                }
            }
            o.resize(n, 0.0);
            lw.wo.matvec(&attn, &mut o)?;
            lw.attn_post_norm.row(0, &mut nbuf)?;
            rmsnorm(&o, &nbuf, cfg.eps, &mut normed)?;
            let attn_out: Vec<f32> = normed.iter().zip(&x).map(|(a, b)| a + b).collect();

            lw.ffn_norm.row(0, &mut nbuf)?;
            rmsnorm(&attn_out, &nbuf, cfg.eps, &mut normed)?;
            lw.w_gate.matvec(&normed, &mut gate)?;
            lw.w_up.matvec(&normed, &mut up)?;
            for i in 0..cfg.n_ffn {
                act[i] = gelu(gate[i]) * up[i];
            }
            lw.w_down.matvec(&act, &mut ffn)?;
            lw.ffn_post_norm.row(0, &mut nbuf)?;
            rmsnorm(&ffn, &nbuf, cfg.eps, &mut normed)?;
            let mut cur: Vec<f32> = normed.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

            // Per-layer embedding (PLE)
            lw.inp_gate.matvec(&cur, &mut ple_gate)?;
            for z in &mut ple_gate {
                *z = gelu(*z);
            }
            let off = il * cfg.n_embd_per_layer;
            for j in 0..cfg.n_embd_per_layer {
                ple_gate[j] *= ple_token[off + j];
            }
            lw.proj.matvec(&ple_gate, &mut ple_out)?;
            lw.post_norm.row(0, &mut nbuf)?;
            rmsnorm(&ple_out, &nbuf, cfg.eps, &mut ple_normed)?;
            for j in 0..n {
                cur[j] += ple_normed[j];
                cur[j] *= lw.out_scale;
            }
            x = cur;
        }

        weights.output_norm.row(0, &mut nbuf)?;
        rmsnorm(&x, &nbuf, cfg.eps, &mut normed)?;
        let mut logits = vec![0f32; cfg.vocab];
        weights.output.matvec(&normed, &mut logits)?;
        if cfg.softcap > 0.0 {
            for v in &mut logits {
                *v = (*v / cfg.softcap).tanh() * cfg.softcap;
            }
        }
        Ok(logits)
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_560_8 * (x + 0.044_715 * x * x * x)).tanh())
}

fn rope_table(n: usize, dim: usize, base: f32, _: Option<()>) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    for p in 0..n {
        for i in 0..half {
            let theta = base.powf(-((2 * i) as f32) / dim as f32);
            let a = p as f32 * theta;
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    (cos, sin)
}

fn learned_rope_table(n: usize, dim: usize, base: f32, freqs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = freqs.len();
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    for p in 0..n {
        for i in 0..half {
            let inv = base.powf(-((2 * i) as f32) / dim as f32);
            // `freqs` are ggml YaRN frequency factors; the effective angle is
            // pos * inv_freq / ff[i].
            let a = p as f32 * inv / freqs[i].max(1e-30);
            cos[p * half + i] = a.cos();
            sin[p * half + i] = a.sin();
        }
    }
    (cos, sin)
}
