//! Greedy generation through the optional llama.cpp CUDA FFI backend.
//!
//!   cargo run --release --example ggml_generate -- MODEL "Once upon a time" 32

use std::env;

use cortex::error::{Error, Result};
use cortex::ggml::GgmlCuda;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        Error::InvalidConfig("usage: ggml_generate <model.gguf> <prompt> <max_tokens>".into())
    })?;
    let prompt = args.get(1).cloned().unwrap_or_else(|| "Hello".into());
    let prompt = if std::env::var_os("CORTEX_GGML_CHAT").is_some() {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        prompt
    };
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mut m = GgmlCuda::load(path, 1024, 4)?;
    if std::env::var_os("GGML_DEBUG").is_some() {
        eprintln!("ids={:?}", m.encode(&prompt)?);
    }
    let ids = m.generate_greedy(&prompt, n)?;
    println!("{}", m.decode_text(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?);
    Ok(())
}
