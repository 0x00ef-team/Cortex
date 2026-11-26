//! Inspect tokenization: `cargo run --example tokens -- model.gguf "some text"`

use cortex::error::{Error, Result};
use cortex::gguf::GgufFile;
use cortex::tokenizer::Tokenizer;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        Error::InvalidConfig("usage: tokens <model.gguf> <text>".into())
    })?;
    let text = args.get(1).cloned().unwrap_or_else(|| "Hello world".into());
    let gguf = GgufFile::load(path)?;
    let t = Tokenizer::from_gguf(&gguf)?;
    let ids = t.encode(&text)?;
    println!("{:?} tokenizer, bos={}, eos={}", t.pre, t.bos(), t.eos());
    println!("ids = {ids:?}");
    for &id in &ids {
        println!(
            "  {id:>6} -> {:?}",
            String::from_utf8_lossy(&t.decode_bytes(id)?)
        );
    }
    println!("decoded = {:?}", t.decode(&ids)?);
    Ok(())
}
