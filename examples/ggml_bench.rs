//! Benchmark the optional llama.cpp CUDA FFI backend.
//!
//!   cargo run --release --example ggml_bench -- MODEL "Once upon a time" 64
//!
//! Prints decode-only tok/s (prefill and load excluded). Requires a prebuilt
//! llama.cpp installation; set CORTEX_GGML_DIR to its `bin` directory.

use std::env;
use std::time::Instant;

use cortex::error::{Error, Result};
use cortex::ggml::GgmlCuda;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| Error::InvalidConfig("usage: ggml_bench <model> <prompt> <n>".into()))?;
    let prompt = args.get(1).cloned().unwrap_or_else(|| "Once upon a time".into());
    let prompt = if std::env::var_os("CORTEX_GGML_CHAT").is_some() {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        prompt
    };
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let context: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let t0 = Instant::now();
    let mut m = GgmlCuda::load(path, context, 4)?;
    let load = t0.elapsed();
    let (n, secs) = m.bench(&prompt, n)?;
    println!("model:      {path}");
    println!("vocab:      {}", m.vocab_size);
    println!("context:    {}", m.context);
    println!("load:       {:.2} s", load.as_secs_f64());
    println!("decode:     {n} tokens in {:.3} s", secs);
    println!("throughput: {:.2} tok/s", n as f64 / secs);
    Ok(())
}
