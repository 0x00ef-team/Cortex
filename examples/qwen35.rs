//! Run a Qwen3.5 hybrid (SSM + attention) GGUF model on CPU.
//!
//!   cargo run --release --example qwen35 -- model.gguf "Hello" --max 8

use std::io::Write;
use std::time::Instant;

use cortex::error::{Error, Result};
use cortex::gguf::GgufFile;
use cortex::{Qwen35, Sampler};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| Error::InvalidConfig("usage: qwen35 <model> <prompt>".into()))?;
    let prompt = args.get(1).cloned().unwrap_or_else(|| "Hello".into());
    let max_tokens: usize = args
        .windows(2)
        .find(|w| w[0] == "--max")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(8);

    let t0 = Instant::now();
    let gguf = GgufFile::load(path)?;
    let tokenizer = Qwen35::tokenizer(&gguf)?;
    drop(gguf);
    let mut model = Qwen35::load(path)?;
    eprintln!("[qwen35] loaded in {:?}, vocab {}", t0.elapsed(), model.vocab_size());

    let ids = tokenizer.encode(&prompt)?;
    eprintln!("[qwen35] prompt tokens: {:?}", ids);
    let mut sampler = Sampler::greedy(0);
    let mut pos = 0usize;
    let mut logits = Vec::new();
    for &id in &ids {
        logits = model.forward(id, pos)?;
        pos += 1;
        eprintln!("[qwen35] prompt {}/{} ({:?})", pos, ids.len(), t0.elapsed());
    }
    let mut next = sampler.sample(&logits, &[])?;
    let mut generated = 0;
    let start = Instant::now();
    while generated < max_tokens {
        let piece = tokenizer.decode(&[next])?;
        print!("{piece}");
        let _ = std::io::stdout().flush();
        generated += 1;
        logits = model.forward(next, pos)?;
        pos += 1;
        next = sampler.sample(&logits, &[])?;
        eprintln!("[qwen35] token {generated} in {:?}", start.elapsed());
    }
    eprintln!("\n[qwen35] {generated} tokens in {:?}", start.elapsed());
    Ok(())
}
