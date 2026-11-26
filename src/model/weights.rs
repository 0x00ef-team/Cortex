//! Weight loading and storage.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::tensor::{DequantPolicy, Matrix};

use super::config::ModelConfig;

/// Weights of a single transformer block.
pub struct LayerWeights {
    pub attn_norm: Matrix,
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    /// Qwen-style attention biases (absent in LLaMA / Mistral).
    pub bq: Option<Matrix>,
    pub bk: Option<Matrix>,
    pub bv: Option<Matrix>,
    pub wo: Matrix,
    pub ffn_norm: Matrix,
    pub w_gate: Matrix,
    pub w_up: Matrix,
    pub w_down: Matrix,
}

/// All weights of the model.
pub struct ModelWeights {
    pub token_embd: Matrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Matrix,
    pub output: Matrix,
}

fn load_matrix(gguf: &mut GgufFile, name: &str, policy: DequantPolicy) -> Result<Matrix> {
    let (kind, ndim, dims) = {
        let info = gguf.info(name)?;
        (info.kind, info.shape.len(), info.shape.clone())
    };
    let (rows, cols) = match ndim {
        1 => (1, dims[0] as usize),
        2 => (dims[1] as usize, dims[0] as usize),
        n => {
            return Err(Error::ShapeMismatch(format!(
                "tensor {name} has {n} dimensions (expected 1 or 2)"
            )))
        }
    };
    let (map, offset, len) = gguf.tensor_view(name)?;
    Matrix::from_gguf_mapped(kind, rows, cols, map, offset, len, policy)
}

fn load_optional_matrix(
    gguf: &mut GgufFile,
    name: &str,
    policy: DequantPolicy,
) -> Result<Option<Matrix>> {
    if gguf.info(name).is_err() {
        return Ok(None);
    }
    Ok(Some(load_matrix(gguf, name, policy)?))
}

impl ModelWeights {
    /// Read every tensor needed by a LLaMA-style transformer.
    pub fn load(gguf: &mut GgufFile, cfg: &ModelConfig, policy: DequantPolicy) -> Result<Self> {
        let token_embd = load_matrix(gguf, "token_embd.weight", policy)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            layers.push(LayerWeights {
                attn_norm: load_matrix(gguf, &p("attn_norm.weight"), policy)?,
                wq: load_matrix(gguf, &p("attn_q.weight"), policy)?,
                wk: load_matrix(gguf, &p("attn_k.weight"), policy)?,
                wv: load_matrix(gguf, &p("attn_v.weight"), policy)?,
                bq: load_optional_matrix(gguf, &p("attn_q.bias"), policy)?,
                bk: load_optional_matrix(gguf, &p("attn_k.bias"), policy)?,
                bv: load_optional_matrix(gguf, &p("attn_v.bias"), policy)?,
                wo: load_matrix(gguf, &p("attn_output.weight"), policy)?,
                ffn_norm: load_matrix(gguf, &p("ffn_norm.weight"), policy)?,
                w_gate: load_matrix(gguf, &p("ffn_gate.weight"), policy)?,
                w_up: load_matrix(gguf, &p("ffn_up.weight"), policy)?,
                w_down: load_matrix(gguf, &p("ffn_down.weight"), policy)?,
            });
        }

        let final_norm = load_matrix(gguf, "output_norm.weight", policy)?;
        let output = match gguf.info("output.weight") {
            Ok(_) => load_matrix(gguf, "output.weight", policy)?,
            Err(_) => {
                // Tied embeddings: reuse the same tensor descriptor.
                load_matrix(gguf, "token_embd.weight", policy)?
            }
        };

        Ok(Self {
            token_embd,
            layers,
            final_norm,
            output,
        })
    }
}
