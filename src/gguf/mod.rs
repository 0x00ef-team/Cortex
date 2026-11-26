//! Minimal GGUF v3 reader.
//!
//! GGUF is the binary container used by llama.cpp.  We only need its header,
//! metadata key/value store and tensor descriptors; tensor data is read
//! lazily directly from the file.

mod reader;
mod value;

pub use reader::{GgufFile, TensorInfo};
pub use value::Value;
