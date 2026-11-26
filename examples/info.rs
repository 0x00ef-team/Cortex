//! Inspect a GGUF file: metadata, tensor shapes, types and parameter counts.

use cortex::error::{Error, Result};
use cortex::gguf::GgufFile;
use cortex::quant::{ggml_type, type_name};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        Error::InvalidConfig("usage: info <model.gguf>".into())
    })?;
    let gguf = GgufFile::load(path)?;

    println!("GGUF version {}", gguf.version);
    println!("\nMetadata:");
    let mut keys: Vec<_> = gguf.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &gguf.metadata[k];
        let preview = match v.str() {
            Some(s) if s.len() < 120 => format!("\"{s}\""),
            _ => format!("{v:?}"),
        };
        println!("  {k} = {}", preview.chars().take(160).collect::<String>());
    }

    println!("\nTensors ({}):", gguf.tensors.len());
    let mut total = 0u64;
    let mut by_type = std::collections::BTreeMap::new();
    for t in &gguf.tensors {
        let elements: u64 = t.shape.iter().product();
        total += elements;
        *by_type.entry(t.kind).or_insert(0u64) += elements;
        println!(
            "  {:<42} {:>12} elements  {:>6}  {:?}",
            t.name, elements, type_name(t.kind), t.shape
        );
    }

    println!("\nParameters: {total}");
    println!("By type:");
    for (kind, n) in by_type {
        let bytes = match kind {
            ggml_type::F32 => 4,
            ggml_type::F16 | ggml_type::BF16 => 2,
            _ => 0,
        };
        let extra = if bytes > 0 {
            format!(" ({:.2} GiB)", n as f64 * bytes as f64 / (1 << 30) as f64)
        } else {
            String::new()
        };
        println!("  {:<8} {n:>12}{extra}", type_name(kind));
    }
    Ok(())
}
