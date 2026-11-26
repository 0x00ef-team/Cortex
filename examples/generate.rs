//! Generate text from a GGUF model.
//!
//! Usage:
//!   cargo run --release --example generate -- model.gguf "Hello" [--temp 0.8] [--max 128] [--seed 42] [--eager]

use std::env;
use std::io::Write;

use cortex::{Engine, Error, GenerateConfig, LoadOptions};

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: generate <model.gguf> <prompt> [--temp N] [--max N] [--top-k N] [--top-p N] [--seed N] [--eager]"
        );
        std::process::exit(2);
    }

    let path = &args[0];
    let prompt = args[1].clone();
    let mut cfg = GenerateConfig::default();
    let mut eager = false;
    let mut threads = 0usize;
    let mut cuda = false;
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--eager" {
            eager = true;
            i += 1;
            continue;
        }
        if flag == "--device" {
            if i + 1 >= args.len() {
                return Err(Error::InvalidConfig("missing value for --device".into()));
            }
            cuda = args[i + 1] == "cuda";
            i += 2;
            continue;
        }
        if flag == "--threads" {
            if i + 1 >= args.len() {
                return Err(Error::InvalidConfig("missing value for --threads".into()));
            }
            threads = args[i + 1].parse().unwrap_or(0);
            i += 2;
            continue;
        }
        if i + 1 >= args.len() {
            return Err(Error::InvalidConfig(format!("missing value for {flag}")));
        }
        let value = &args[i + 1];
        match flag {
            "--temp" => cfg.temperature = value.parse().unwrap_or(0.8),
            "--max" => cfg.max_tokens = value.parse().unwrap_or(128),
            "--top-k" => cfg.top_k = value.parse().unwrap_or(40),
            "--top-p" => cfg.top_p = value.parse().unwrap_or(0.9),
            "--seed" => cfg.seed = value.parse().unwrap_or(0),
            other => return Err(Error::InvalidConfig(format!("unknown flag {other}"))),
        }
        i += 2;
    }

    let options = LoadOptions {
        dequant: if eager {
            cortex::DequantPolicy::Eager
        } else {
            cortex::DequantPolicy::Lazy
        },
        threads,
        device: if cuda {
            cortex::Device::Cuda(0)
        } else {
            cortex::Device::Cpu
        },
        progress: Some(Box::new(|msg| eprintln!("[cortex] {msg}"))),
        ..Default::default()
    };

    let mut engine = Engine::load(path, options)?;
    let out = engine.generate(&prompt, &cfg, |_, piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;
    eprintln!("\n\n{} tokens generated", out.token_count);
    Ok(())
}
