//! Model hyperparameters read from GGUF metadata.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;

/// How rotary position embeddings pair up channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    /// LLaMA / Mistral: pairs `(2i, 2i+1)`.
    Llama,
    /// GPT-NeoX / Qwen: pairs `(i, i + dim/2)`.
    NeoX,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_embd: usize,
    pub n_ffn: usize,
    pub head_dim: usize,
    pub n_ctx: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_style: RopeStyle,
    pub rope_dim: usize,
    /// Linear scaling factor applied to positions before RoPE.
    pub rope_scale: f32,
}

impl ModelConfig {
    /// Parse the metadata of a GGUF file into a model configuration.
    pub fn from_gguf(gguf: &GgufFile, vocab_size: usize) -> Result<Self> {
        let arch = gguf
            .get_str("general.architecture")
            .ok_or_else(|| Error::InvalidConfig("missing general.architecture".into()))?;
        let p = |suffix: &str| format!("{arch}.{suffix}");

        let n_layers = gguf
            .get_u32(&p("block_count"))
            .ok_or_else(|| Error::InvalidConfig("missing block_count".into()))?
            as usize;
        let n_embd = gguf
            .get_u32(&p("embedding_length"))
            .ok_or_else(|| Error::InvalidConfig("missing embedding_length".into()))?
            as usize;
        let n_ffn = gguf
            .get_u32(&p("feed_forward_length"))
            .unwrap_or((n_embd * 4) as u32) as usize;
        let n_heads = gguf
            .get_u32(&p("attention.head_count"))
            .ok_or_else(|| Error::InvalidConfig("missing attention.head_count".into()))?
            as usize;
        let n_kv_heads = gguf
            .get_u32(&p("attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let n_ctx = gguf.get_u32(&p("context_length")).unwrap_or(4096) as usize;
        let rms_eps = gguf
            .get_f32(&p("attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_f32(&p("layer_norm_rms_epsilon")))
            .unwrap_or(1e-5);
        let rope_base = gguf
            .get_f32(&p("attention.rope.freq_base"))
            .or_else(|| gguf.get_f32(&p("rope.freq_base")))
            .unwrap_or(10_000.0);
        let head_dim = gguf
            .get_u32(&p("attention.key_length"))
            .unwrap_or((n_embd / n_heads) as u32) as usize;
        let rope_dim = gguf
            .get_u32(&p("rope.dimension_count"))
            .unwrap_or(head_dim as u32) as usize;

        if n_layers == 0 || n_embd == 0 || n_heads == 0 || n_kv_heads == 0 {
            return Err(Error::InvalidConfig("zero-sized model dimension".into()));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(Error::InvalidConfig(
                "head_count is not divisible by head_count_kv".into(),
            ));
        }
        if head_dim == 0 || rope_dim == 0 || rope_dim % 2 != 0 {
            return Err(Error::InvalidConfig("invalid rotary dimension".into()));
        }
        if head_dim * n_heads != n_embd {
            return Err(Error::InvalidConfig(format!(
                "head_dim*head_count ({}) != embedding_length ({n_embd})",
                head_dim * n_heads
            )));
        }

        let rope_style = if arch.contains("qwen") || arch.contains("neo") || arch.contains("phi") {
            RopeStyle::NeoX
        } else {
            RopeStyle::Llama
        };
        let rope_scale = gguf
            .get_f32(&p("rope.scaling.factor"))
            .or_else(|| gguf.get_f32(&p("attention.rope.scaling.factor")))
            .unwrap_or(1.0);

        Ok(Self {
            architecture: arch,
            vocab_size,
            n_layers,
            n_heads,
            n_kv_heads,
            n_embd,
            n_ffn,
            head_dim,
            n_ctx,
            rms_eps,
            rope_base,
            rope_style,
            rope_dim,
            rope_scale,
        })
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}
