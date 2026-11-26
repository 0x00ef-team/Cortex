//! Attention-related numeric kernels.

use crate::tensor::dot;
use crate::Result;

/// RMS normalization: `out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)`.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
    if x.len() != w.len() || out.len() != x.len() {
        return Err(crate::error::Error::ShapeMismatch(
            "rmsnorm operand sizes differ".into(),
        ));
    }
    let ss = dot(x, x) / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = w[i] * (scale * x[i]);
    }
    Ok(())
}

/// Stable softmax over a slice, in place.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in &x[1..] {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// SiLU (a.k.a. Swish) used by the SwiGLU feed-forward network.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `out = silu(gate) * up`
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || out.len() != gate.len() {
        return Err(crate::error::Error::ShapeMismatch(
            "swiglu operand sizes differ".into(),
        ));
    }
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

/// Precompute cos/sin tables of shape `[n_positions, dim/2]`.
pub fn rope_table(
    n_positions: usize,
    dim: usize,
    base: f32,
    _style: super::RopeStyle,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; n_positions * half];
    let mut sin = vec![0.0f32; n_positions * half];
    for pos in 0..n_positions {
        for i in 0..half {
            // Both styles use frequencies base^(-2i/dim); they differ only
            // in how channels are paired for rotation.
            let channel = 2 * i;
            let theta = base.powf(-(channel as f32) / dim as f32);
            let angle = (pos as f32) * scale * theta;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// Apply rotary embedding to one head inside `x`.
///
/// `cos`/`sin` are the row for the current position, of length `dim/2`.
pub fn rotate_half(
    x: &mut [f32],
    offset: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
    style: super::RopeStyle,
) {
    let half = dim / 2;
    for i in 0..half {
        let (a, b) = match style {
            super::RopeStyle::Llama => (offset + 2 * i, offset + 2 * i + 1),
            super::RopeStyle::NeoX => (offset + i, offset + i + half),
        };
        let (va, vb) = (x[a], x[b]);
        let (c, s) = (cos[i], sin[i]);
        x[a] = va * c - vb * s;
        x[b] = vb * c + va * s;
    }
}
