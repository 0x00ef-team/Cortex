//! Error types used throughout the engine.

use std::fmt;

/// Errors produced by loading, decoding and running a model.
#[derive(Debug)]
pub enum Error {
    /// File system / IO failure.
    Io(std::io::Error),
    /// The file is not a valid GGUF file or has an unexpected layout.
    InvalidGguf(String),
    /// A GGUF tensor uses a quantization type we do not implement.
    UnsupportedType(u32),
    /// A tensor required by the architecture is missing.
    MissingTensor(String),
    /// Two tensors disagree about a dimension, or a shape is invalid.
    ShapeMismatch(String),
    /// Metadata required to build the model is missing or malformed.
    InvalidConfig(String),
    /// Tokenizer data is inconsistent.
    InvalidTokenizer(String),
    /// An index produced by the model is outside the vocabulary.
    InvalidToken(u32),
    /// The KV cache ran out of context space.
    ContextOverflow { pos: usize, capacity: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::InvalidGguf(s) => write!(f, "invalid GGUF: {s}"),
            Error::UnsupportedType(t) => write!(f, "unsupported GGML tensor type {t}"),
            Error::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Error::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
            Error::InvalidConfig(s) => write!(f, "invalid model config: {s}"),
            Error::InvalidTokenizer(s) => write!(f, "invalid tokenizer: {s}"),
            Error::InvalidToken(t) => write!(f, "invalid token id: {t}"),
            Error::ContextOverflow { pos, capacity } => {
                write!(f, "context overflow at position {pos} (capacity {capacity})")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
