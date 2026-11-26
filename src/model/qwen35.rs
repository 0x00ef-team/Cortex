//! Qwen3.5 hybrid (SSM + full attention) inference, CPU path.
//!
//! Math ported from llama.cpp `src/models/qwen35.cpp`. See issue #6 for the
//! reference notes. This model family interleaves 3 linear-attention (GDN)
//! layers with 1 full-attention layer.

use std::path::Path;

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::model::attention::{rmsnorm, softmax};
use crate::tensor::{dot, DequantPolicy, Matrix};
use crate::tokenizer::Tokenizer;

const D: usize = 4096;
const FFN: usize = 12288;
const LAYERS: usize = 32;
const N_HEAD: usize = 16;
const N_KV: usize = 4;
const HD: usize = 256;        // full-attention head dim
const GDN_HD: usize = 128;    // linear-attention head dim
const GDN_V: usize = 32;      // value heads
const STATE: usize = 128;
const CONV_CH: usize = 8192;
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
enum LayerType {
    Ssm,
    Full,
}

struct AttentionLayer {
    norm: Matrix,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: Matrix,
    k_norm: Matrix,
    post_norm: Matrix,
}

struct SsmLayer {
    norm: Matrix,
    qkv: Matrix,
    gate: Matrix,
    a: Matrix,
    alpha: Matrix,
    beta: Matrix,
    dt: Matrix,
    conv1d: Matrix,
    ssm_norm: Matrix,
    out: Matrix,
    post_norm: Matrix,
}

enum Branch {
    Attn(AttentionLayer),
    Ssm(SsmLayer),
}

struct Layer {
    branch: Branch,
    ffn_gate: Matrix,
    ffn_up: Matrix,
    ffn_down: Matrix,
}

struct Weights {
    embd: Matrix,
    layers: Vec<Layer>,
    final_norm: Matrix,
    output: Matrix,
}

fn load_mat(g: &GgufFile, name: &str) -> Result<Matrix> {
    let info = g.info(name)?;
    let (map, off, len) = g.tensor_view(name)?;
    Matrix::from_gguf_mapped(
        info.kind,
        info.rows(),
        info.cols(),
        map,
        off,
        len,
        DequantPolicy::Lazy,
    )
}

fn load_vec(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let m = load_mat(g, name)?;
    let mut v = vec![0f32; m.cols()];
    m.row(0, &mut v)?;
    Ok(v)
}

impl Weights {
    fn load(g: &GgufFile) -> Result<Self> {
        let mut layers = Vec::with_capacity(LAYERS);
        for il in 0..LAYERS {
            let p = |s: &str| format!("blk.{il}.{s}");
            let full = g.info(&p("attn_q.weight")).is_ok();
            let branch = if full {
                Branch::Attn(AttentionLayer {
                    norm: load_mat(g, &p("attn_norm.weight"))?,
                    wq: load_mat(g, &p("attn_q.weight"))?,
                    wk: load_mat(g, &p("attn_k.weight"))?,
                    wv: load_mat(g, &p("attn_v.weight"))?,
                    wo: load_mat(g, &p("attn_output.weight"))?,
                    q_norm: load_mat(g, &p("attn_q_norm.weight"))?,
                    k_norm: load_mat(g, &p("attn_k_norm.weight"))?,
                    post_norm: load_mat(g, &p("post_attention_norm.weight"))?,
                })
            } else {
                Branch::Ssm(SsmLayer {
                    norm: load_mat(g, &p("attn_norm.weight"))?,
                    qkv: load_mat(g, &p("attn_qkv.weight"))?,
                    gate: load_mat(g, &p("attn_gate.weight"))?,
                    a: load_mat(g, &p("ssm_a"))?,
                    alpha: load_mat(g, &p("ssm_alpha.weight"))?,
                    beta: load_mat(g, &p("ssm_beta.weight"))?,
                    dt: load_mat(g, &p("ssm_dt.bias"))?,
                    conv1d: load_mat(g, &p("ssm_conv1d.weight"))?,
                    ssm_norm: load_mat(g, &p("ssm_norm.weight"))?,
                    out: load_mat(g, &p("ssm_out.weight"))?,
                    post_norm: load_mat(g, &p("post_attention_norm.weight"))?,
                })
            };
            layers.push(Layer {
                branch,
                ffn_gate: load_mat(g, &p("ffn_gate.weight"))?,
                ffn_up: load_mat(g, &p("ffn_up.weight"))?,
                ffn_down: load_mat(g, &p("ffn_down.weight"))?,
            });
        }
        Ok(Self {
            embd: load_mat(g, "token_embd.weight")?,
            layers,
            final_norm: load_mat(g, "output_norm.weight")?,
            output: load_mat(g, "output.weight")?,
        })
    }
}

/// Recurrent state of one GDN layer.
struct SsmState {
    /// [3, CONV_CH] rolling convolution history.
    conv_hist: Vec<f32>,
    /// [GDN_V, STATE, STATE] recurrent matrices.
    states: Vec<f32>,
}

impl SsmState {
    fn new() -> Self {
        Self {
            conv_hist: vec![0.0; 3 * CONV_CH],
            states: vec![0.0; GDN_V * STATE * STATE],
        }
    }
}

/// Qwen3.5 model ready for autoregressive decoding on CPU.
pub struct Qwen35 {
    weights: Weights,
    vocab: usize,
    kv_keys: Vec<Vec<f32>>,
    kv_values: Vec<Vec<f32>>,
    ssm: Vec<SsmState>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Qwen35 {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let g = GgufFile::load(path)?;
        let vocab = g
            .get_u32("qwen35.vocab_size")
            .or_else(|| {
                g.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() as u32)
            })
            .ok_or_else(|| Error::InvalidConfig("cannot determine vocab size".into()))?
            as usize;
        let weights = Weights::load(&g)?;
        let n_attn = (0..LAYERS)
            .filter(|&il| g.info(&format!("blk.{il}.attn_q.weight")).is_ok())
            .count();
        let _ = n_attn;
        let kv_keys = vec![Vec::new(); LAYERS];
        let kv_values = vec![Vec::new(); LAYERS];
        let ssm = (0..LAYERS).map(|_| SsmState::new()).collect();

        // M-RoPE with identical text positions reduces to a per-pair frequency
        // table over the first 64 head dimensions.
        let base = g.get_f32("qwen35.rope.freq_base").unwrap_or(1e7);
        let dim = g.get_u32("qwen35.rope.dimension_count").unwrap_or(64) as usize;
        let half = dim / 2;
        let n_pos = 4096usize;
        let mut rope_cos = vec![0f32; n_pos * half];
        let mut rope_sin = vec![0f32; n_pos * half];
        for p in 0..n_pos {
            for s in 0..half {
                let theta = base.powf(-(s as f32) / half as f32);
                let a = p as f32 * theta;
                rope_cos[p * half + s] = a.cos();
                rope_sin[p * half + s] = a.sin();
            }
        }
        Ok(Self {
            weights,
            vocab,
            kv_keys,
            kv_values,
            ssm,
            rope_cos,
            rope_sin,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn reset_cache(&mut self) {
        for k in &mut self.kv_keys { k.clear(); }
        for v in &mut self.kv_values { v.clear(); }
        for s in &mut self.ssm {
            s.conv_hist.fill(0.0);
            s.states.fill(0.0);
        }
    }

    pub fn tokenizer(gguf: &GgufFile) -> Result<Tokenizer> {
        Tokenizer::from_gguf(gguf)
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        if token as usize >= self.vocab {
            return Err(Error::InvalidToken(token));
        }
        let w = &self.weights;
        let mut x = vec![0f32; D];
        w.embd.row(token as usize, &mut x)?;

        for il in 0..LAYERS {
            let layer = &w.layers[il];
            let mut a = vec![0f32; D];
            match &layer.branch {
                Branch::Attn(l) => {
                    let mut nb = vec![0f32; D];
                    l.norm.row(0, &mut nb)?;
                    rmsnorm(&x, &nb, EPS, &mut a)?;
                    let keys = &mut self.kv_keys[il];
                    let vals = &mut self.kv_values[il];
                    let y = Self::forward_attention(l, &a, keys, vals, &self.rope_cos, &self.rope_sin, pos)?;
                    for i in 0..D { x[i] += y[i]; }
                }
                Branch::Ssm(l) => {
                    let mut nb = vec![0f32; D];
                    l.norm.row(0, &mut nb)?;
                    rmsnorm(&x, &nb, EPS, &mut a)?;
                    let y = Self::forward_ssm(l, &mut self.ssm[il], &a)?;
                    for i in 0..D { x[i] += y[i]; }
                }
            }
            let mut nb = vec![0f32; D];
            match &layer.branch {
                Branch::Attn(l) => l.post_norm.row(0, &mut nb)?,
                Branch::Ssm(l) => l.post_norm.row(0, &mut nb)?,
            }
            let mut f = vec![0f32; D];
            rmsnorm(&x, &nb, EPS, &mut f)?;
            let mut g = vec![0f32; FFN];
            let mut u = vec![0f32; FFN];
            layer.ffn_gate.matvec(&f, &mut g)?;
            layer.ffn_up.matvec(&f, &mut u)?;
            for i in 0..FFN { g[i] = silu(g[i]) * u[i]; }
            let mut d = vec![0f32; D];
            layer.ffn_down.matvec(&g, &mut d)?;
            for i in 0..D { x[i] += d[i]; }
        }

        let mut nb = vec![0f32; D];
        w.final_norm.row(0, &mut nb)?;
        let mut h = vec![0f32; D];
        rmsnorm(&x, &nb, EPS, &mut h)?;
        let mut logits = vec![0f32; self.vocab];
        w.output.matvec(&h, &mut logits)?;
        Ok(logits)
    }

    fn forward_attention(
        l: &AttentionLayer,
        a: &[f32],
        keys: &mut Vec<f32>,
        vals: &mut Vec<f32>,
        cos_table: &[f32],
        sin_table: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>> {
        const QDIM: usize = N_HEAD * HD;
        const KVDIM: usize = N_KV * HD;
        let mut qg = vec![0f32; QDIM * 2];
        let mut k = vec![0f32; KVDIM];
        let mut v = vec![0f32; KVDIM];
        l.wq.matvec(a, &mut qg)?;
        l.wk.matvec(a, &mut k)?;
        l.wv.matvec(a, &mut v)?;
        let mut qn = vec![0f32; HD];
        l.q_norm.row(0, &mut qn)?;
        for h in 0..N_HEAD {
            let mut head = vec![0f32; HD];
            head.copy_from_slice(&qg[h * HD..(h + 1) * HD]);
            let mut normed = vec![0f32; HD];
            rmsnorm(&head, &qn, EPS, &mut normed)?;
            head.copy_from_slice(&normed);
            let half = cos_table.len() / 4096;
            rotate(
                &mut qg, h * HD, &head,
                &cos_table[pos * half..(pos + 1) * half],
                &sin_table[pos * half..(pos + 1) * half],
            );
        }
        let mut kn = vec![0f32; HD];
        l.k_norm.row(0, &mut kn)?;
        for h in 0..N_KV {
            let mut head = vec![0f32; HD];
            head.copy_from_slice(&k[h * HD..(h + 1) * HD]);
            let mut normed = vec![0f32; HD];
            rmsnorm(&head, &kn, EPS, &mut normed)?;
            head.copy_from_slice(&normed);
            let half = cos_table.len() / 4096;
            rotate(
                &mut k, h * HD, &head,
                &cos_table[pos * half..(pos + 1) * half],
                &sin_table[pos * half..(pos + 1) * half],
            );
        }

        keys.extend_from_slice(&k);
        vals.extend_from_slice(&v);
        let n = keys.len() / KVDIM;

        let mut o = vec![0f32; QDIM];
        let scale = 1.0 / (HD as f32).sqrt();
        let mut scores = vec![0f32; n];
        for h in 0..N_HEAD {
            let kv_head = h / (N_HEAD / N_KV);
            let qh = &qg[h * HD..(h + 1) * HD];
            for t in 0..n {
                let kh = &keys[t * KVDIM + kv_head * HD..t * KVDIM + kv_head * HD + HD];
                scores[t] = dot(qh, kh) * scale;
            }
            softmax(&mut scores);
            let oh = &mut o[h * HD..(h + 1) * HD];
            oh.fill(0.0);
            for t in 0..n {
                let vh = &vals[t * KVDIM + kv_head * HD..t * KVDIM + kv_head * HD + HD];
                let a = scores[t];
                for j in 0..HD { oh[j] += a * vh[j]; }
            }
        }
        // Attention gate (second half of qg) gates the output per feature.
        for i in 0..QDIM {
            let g = sigmoid(qg[QDIM + i]);
            o[i] *= g;
        }
        let mut y = vec![0f32; D];
        l.wo.matvec(&o, &mut y)?;
        Ok(y)
    }

    fn forward_ssm(l: &SsmLayer, st: &mut SsmState, a: &[f32]) -> Result<Vec<f32>> {
        let mut qkv = vec![0f32; CONV_CH];
        let mut z = vec![0f32; D];
        l.qkv.matvec(a, &mut qkv)?;
        l.gate.matvec(a, &mut z)?;

        let mut beta = vec![0f32; GDN_V];
        let mut alpha = vec![0f32; GDN_V];
        l.beta.matvec(a, &mut beta)?;
        l.alpha.matvec(a, &mut alpha)?;
        let mut dt = vec![0f32; GDN_V];
        l.dt.row(0, &mut dt)?;
        let mut aa = vec![0f32; GDN_V];
        l.a.row(0, &mut aa)?;
        for i in 0..GDN_V {
            beta[i] = sigmoid(beta[i]);
            alpha[i] += dt[i];
            alpha[i] = aa[i] * softplus(alpha[i]);
        }

        // Depthwise causal convolution with rolling history.
        let mut conv = vec![0f32; CONV_CH];
        let mut conv_w = vec![0f32; 4];
        for c in 0..CONV_CH {
            l.conv1d.row(c, &mut conv_w)?;
            let s = st.conv_hist[2 * CONV_CH + c];
            let s2 = st.conv_hist[CONV_CH + c];
            let s3 = st.conv_hist[c];
            conv[c] = silu(s * conv_w[0] + s2 * conv_w[1] + s3 * conv_w[2] + qkv[c] * conv_w[3]);
        }
        // Shift history: drop oldest, append current qkv.
        st.conv_hist.copy_within(CONV_CH..3 * CONV_CH, 0);
        st.conv_hist[2 * CONV_CH..].copy_from_slice(&qkv);

        // Split q/k (16 heads of 128) and v (32 heads of 128).
        let q_off = 0usize;
        let k_off = N_HEAD * GDN_HD;
        let v_off = 2 * N_HEAD * GDN_HD;
        for h in 0..N_HEAD {
            l2norm(&mut conv[q_off + h * GDN_HD..q_off + (h + 1) * GDN_HD]);
            l2norm(&mut conv[k_off + h * GDN_HD..k_off + (h + 1) * GDN_HD]);
        }

        let mut nrm = vec![0f32; STATE];
        l.ssm_norm.row(0, &mut nrm)?;
        let mut y = vec![0f32; D];
        for hv in 0..GDN_V {
            let hk = hv % N_HEAD;
            let qh = &conv[q_off + hk * GDN_HD..q_off + hk * GDN_HD + GDN_HD];
            let kh = &conv[k_off + hk * GDN_HD..k_off + hk * GDN_HD + GDN_HD];
            let vh = &conv[v_off + hv * GDN_HD..v_off + hv * GDN_HD + GDN_HD];
            let s = &mut st.states[hv * STATE * STATE..(hv + 1) * STATE * STATE];
            let decay = alpha[hv].exp();
            for row in s.iter_mut() { *row *= decay; }
            let mut d = vec![0f32; STATE];
            for j in 0..STATE {
                let dot_sv = dot(&s[j * STATE..(j + 1) * STATE], kh);
                d[j] = beta[hv] * (vh[j] - dot_sv);
            }
            for j in 0..STATE {
                let base = j * STATE;
                let dj = d[j];
                for i in 0..STATE { s[base + i] += kh[i] * dj; }
            }
            let inv = 1.0 / (STATE as f32).sqrt();
            for j in 0..STATE {
                let o = dot(&s[j * STATE..(j + 1) * STATE], qh) * inv;
                // per-head RMS normalization with ssm_norm weights
                let rms = (o * o / STATE as f32 + EPS).sqrt().recip();
                y[hv * GDN_HD + j] = o * rms * nrm[j] * silu(z[hv * GDN_HD + j]);
            }
        }
        let mut out = vec![0f32; D];
        l.out.matvec(&y, &mut out)?;
        Ok(out)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
#[inline]
fn l2norm(v: &mut [f32]) {
    let ss = dot(v, v);
    let s = 1.0 / (ss + EPS).sqrt();
    for x in v.iter_mut() { *x *= s; }
}

/// Rotate the first 64 channels of a 256-dim head using M-Rope frequencies.
fn rotate(
    head: &mut [f32],
    offset: usize,
    normalized: &[f32],
    cos: &[f32],
    sin: &[f32],
) {
    for i in 0..normalized.len() { head[offset + i] = normalized[i]; }
    let half = cos.len();
    for i in 0..half {
        let a = head[offset + i];
        let b = head[offset + i + half];
        head[offset + i] = a * cos[i] - b * sin[i];
        head[offset + i + half] = b * cos[i] + a * sin[i];
    }
}
