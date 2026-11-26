//! # Cortex
//!
//! A from-scratch inference engine for LLaMA-style language models, written
//! in Rust with no external dependencies.
//!
//! ```no_run
//! use cortex::{Engine, GenerateConfig};
//!
//! let mut engine = Engine::load("model.gguf", Default::default())?;
//! let out = engine.generate("Hello", &GenerateConfig::default(), |_, piece| {
//!     print!("{piece}");
//! })?;
//! println!("\n{} tokens", out.token_count);
//! # Ok::<(), cortex::Error>(())
//! ```
//!
//! Implemented from first principles:
//! - GGUF v3 container parsing and metadata
//! - GGML dequantization (F32/F16/BF16, Q4/Q5/Q8, K-quants Q2..Q6)
//! - Matrix-vector kernels, RMSNorm, RoPE, SwiGLU and softmax
//! - Multi-head / grouped-query attention with a growing KV cache
//! - BPE tokenization driven by GGUF vocabularies
//! - Optional CUDA backend: JIT-compiled `matvec` kernel, weights in VRAM
//! - Temperature, top-k, top-p, min-p and repetition-penalty sampling

#[allow(unused_unsafe)]
pub mod cuda;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod model;
pub mod quant;
pub mod sampler;
pub mod synthetic;
pub mod tensor;
pub mod thread_pool;
pub mod tokenizer;

pub use error::{Error, Result};
pub use cuda::CudaDevice;
pub use generate::{generate, GenerateConfig, GenerateOutput};
pub use model::{Llama, LoadOptions, ModelConfig, RopeStyle};
pub use sampler::{Rng, Sampler};
pub use tensor::DequantPolicy;
pub use thread_pool::ThreadPool;
pub use tokenizer::{PreTokenizer, Tokenizer};

/// Compute device used for matrix-vector products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Portable CPU backend (default).
    Cpu,
    /// NVIDIA GPU with the given device index. Requires the CUDA driver and
    /// CUDA Toolkit (for NVRTC). Weights are expanded to `f32` and uploaded.
    Cuda(u32),
}

/// Convenience wrapper pairing a model with its tokenizer.
pub struct Engine {
    pub model: Llama,
    pub tokenizer: Tokenizer,
}

impl Engine {
    /// Load model weights and tokenizer from a single GGUF file.
    pub fn load(path: impl AsRef<std::path::Path>, options: LoadOptions) -> Result<Self> {
        Self::from_gguf(path, options)
    }

    /// Alias for [`Engine::load`].
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        options: LoadOptions,
    ) -> Result<Self> {
        // Parsing the header twice is cheap compared to loading the weights;
        // it keeps `Llama` independent of tokenizer concerns.
        let gguf = gguf::GgufFile::load(path.as_ref())?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        drop(gguf);
        let model = Llama::load(path, options)?;
        Ok(Self { model, tokenizer })
    }

    /// Generate text, streaming completed UTF-8 fragments to `on_token`.
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(u32, &str),
    {
        generate(
            &mut self.model,
            &self.tokenizer,
            prompt,
            cfg,
            on_token,
        )
    }
}
