//! Minimal Gemma 4 generation example (CPU).
//!
//!   cargo run --release --example gemma4 -- model.gguf "Hello" --max 32

use std::env;
use std::io::Write;
use std::time::Instant;

use cortex::gguf::GgufFile;
use cortex::tokenizer::Tokenizer;
use cortex::{Error, Gemma4, GenerateConfig, Sampler};

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        Error::InvalidConfig("usage: gemma4 <model.gguf> <prompt> [--max N]".into())
    })?;
    let prompt = args.get(1).cloned().unwrap_or_else(|| "Hello".into());
    let max_tokens: usize = args
        .windows(2)
        .find(|w| w[0] == "--max")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(32);
    let cuda = args.iter().any(|a| a == "--cuda");

    let t0 = Instant::now();
    let gguf = GgufFile::load(path)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    drop(gguf);
    let mut model = Gemma4::load_with_device(path, if cuda { Some(0) } else { None })?;
    eprintln!("[gemma4] loaded in {:?}, {} layers", t0.elapsed(), model.config.n_layers);

    model.reset_cache();
    let ids = tokenizer.encode(&prompt)?;
    eprintln!("[gemma4] prompt tokens: {ids:?}");
    let mut logits = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        logits = model.forward(id, pos)?;
    }
    let mut pos = ids.len();
    let cfg = GenerateConfig {
        max_tokens,
        ..Default::default()
    };
    let mut sampler = Sampler::new(cfg.seed);
    sampler.temperature = cfg.temperature;
    sampler.top_k = cfg.top_k;
    sampler.top_p = cfg.top_p;
    let mut generated = Vec::new();
    let mut next = sampler.sample(&logits, &generated)?;
    let t1 = Instant::now();
    for _ in 0..cfg.max_tokens {
        if cfg.stop_on_eos && next == tokenizer.eos() {
            break;
        }
        let piece = tokenizer.decode(&[next])?;
        print!("{piece}");
        let _ = std::io::stdout().flush();
        generated.push(next);
        logits = model.forward(next, pos)?;
        pos += 1;
        next = sampler.sample(&logits, &generated)?;
    }
    eprintln!("\n[gemma4] {} tokens in {:?}", generated.len(), t1.elapsed());
    Ok(())
}
