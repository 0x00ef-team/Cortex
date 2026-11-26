//! Create a tiny synthetic GGUF model for trying out the CLI without a
//! real model: `cargo run --example make_tiny -- /tmp/tiny.gguf`

use cortex::error::Result;
use cortex::synthetic::write_tiny_model;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "/tmp/cortex-tiny.gguf".to_string());
    write_tiny_model(&path, 42)?;
    println!("wrote {path}");
    Ok(())
}
