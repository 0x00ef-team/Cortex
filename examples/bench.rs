//! Benchmark generation throughput.
//!
//!   cargo run --release --example bench -- model.gguf "Once upon a time" 16 [--threads 8] [--eager]
//!
//! Prints prefill and decode speed in tokens/second.

use std::env;
use std::time::Instant;

use cortex::error::{Error, Result};
use cortex::{Engine, GenerateConfig, LoadOptions};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| Error::InvalidConfig("usage: bench <model> <prompt> <n>".into()))?;
    let prompt = args.get(1).cloned().unwrap_or_else(|| "Once upon a time".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut threads = 0usize;
    let mut eager = false;
    let mut cuda = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                threads = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--eager" => {
                eager = true;
                i += 1;
            }
            "--device" => {
                cuda = args.get(i + 1).map(|s| s.as_str()) == Some("cuda");
                i += 2;
            }
            _ => i += 1,
        }
    }

    let t0 = Instant::now();
    let options = LoadOptions {
        threads,
        device: if cuda {
            cortex::Device::Cuda(0)
        } else {
            cortex::Device::Cpu
        },
        dequant: if eager {
            cortex::DequantPolicy::Eager
        } else {
            cortex::DequantPolicy::Lazy
        },
        ..Default::default()
    };
    eprintln!("DBG loading"); let mut engine = Engine::load(path, options)?; eprintln!("DBG loaded");
    let load = t0.elapsed();

    let cfg = GenerateConfig {
        max_tokens: n,
        temperature: 0.0,
        ..Default::default()
    };
    let t1 = Instant::now();
    eprintln!("DBG generating"); let out = engine.generate(&prompt, &cfg, |_, _| {})?; eprintln!("DBG generated");
    let total = t1.elapsed();

    println!("model:        {path}");
    println!("load:         {:.2} s", load.as_secs_f64());
    println!("generated:    {} tokens in {:.2} s", out.token_count, total.as_secs_f64());
    println!("throughput:   {:.2} tok/s (prefill included)",
        out.token_count as f64 / total.as_secs_f64());
    println!("threads:      {}", engine.model.threads());
    if let Some((ns, calls)) = engine.model.cuda_stats() {
        println!("GPU matvec:   {:.2} ms total / {} calls ({:.0} us/call)",
            ns as f64 / 1e6, calls, ns as f64 / calls.max(1) as f64 / 1e3);
        let mut by: Vec<_> = engine
            .model
            .cuda_stats_by_shape()
            .unwrap_or_default()
            .into_iter()
            .collect();
        by.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for ((rows, cols), (ns, calls)) in by.into_iter().take(6) {
            println!("  {rows:>6}x{cols:<6} {:.1} ms / {} calls ({:.0} us/call)",
                ns as f64 / 1e6, calls, ns as f64 / calls.max(1) as f64 / 1e3);
        }
    }
    Ok(())
}
