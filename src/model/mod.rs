//! Transformer model implementation.

mod attention;
mod cache;
mod config;
mod gemma4;
mod gpu;
mod llama;
mod weights;

pub use cache::LayerCache;
pub use config::{ModelConfig, RopeStyle};
pub use gemma4::{Gemma4, Gemma4Config};
pub use llama::{Llama, LoadOptions};
pub use weights::{LayerWeights, ModelWeights};

pub use attention::{rmsnorm, rope_table, softmax, swiglu};
