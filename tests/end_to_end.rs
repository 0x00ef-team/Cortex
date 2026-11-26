//! End-to-end tests using a generated tiny GGUF model.

use std::path::PathBuf;

use cortex::gguf::GgufFile;
use cortex::model::RopeStyle;
use cortex::synthetic::write_tiny_model;
use cortex::tokenizer::Tokenizer;
use cortex::{Engine, Error, GenerateConfig, LoadOptions};

fn tiny_model(seed: u64) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cortex-tiny-{}-{seed}.gguf",
        std::process::id()
    ));
    write_tiny_model(&path, seed).unwrap();
    path
}

#[test]
fn gguf_header_and_metadata() {
    let path = tiny_model(1);
    let mut gguf = GgufFile::load(&path).unwrap();
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.get_str("general.architecture").as_deref(), Some("llama"));
    assert_eq!(gguf.get_u32("llama.block_count"), Some(2));
    assert_eq!(gguf.get_u32("llama.embedding_length"), Some(16));
    assert_eq!(gguf.tensors.len(), 21);
    let emb = gguf.info("token_embd.weight").unwrap();
    assert_eq!(emb.shape, vec![16, 32]);
    assert_eq!(emb.rows(), 32);
    assert_eq!(emb.cols(), 16);
    let bytes = gguf.tensor_bytes("token_embd.weight").unwrap();
    assert_eq!(bytes.len(), 32 * 16 * 4);
}

#[test]
fn tokenizer_roundtrip() {
    let path = tiny_model(2);
    let gguf = GgufFile::load(&path).unwrap();
    let t = Tokenizer::from_gguf(&gguf).unwrap();
    assert_eq!(t.vocab_size(), 32);
    assert_eq!(t.bos(), 1);
    assert_eq!(t.eos(), 2);

    let ids = t.encode("ab").unwrap();
    assert_eq!(ids.first(), Some(&1));
    let decoded = t.decode(&ids).unwrap();
    assert!(decoded.contains('a'));
    assert!(decoded.contains('b'));
}

#[test]
fn forward_produces_finite_logits() {
    let path = tiny_model(3);
    let mut engine = Engine::load(&path, Default::default()).unwrap();
    assert_eq!(engine.model.config.n_layers, 2);
    assert_eq!(engine.model.config.n_kv_heads, 2);
    assert_eq!(engine.model.config.rope_style, RopeStyle::Llama);
    assert_eq!(engine.model.context_length(), 32);

    let logits = engine.model.forward(1, 0).unwrap();
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn generation_is_deterministic_and_resettable() {
    let path = tiny_model(4);
    let mut engine = Engine::load(&path, Default::default()).unwrap();
    let cfg = GenerateConfig {
        max_tokens: 8,
        temperature: 0.7,
        seed: 123,
        ..Default::default()
    };
    let a = engine.generate("a", &cfg, |_, _| {}).unwrap();
    let b = engine.generate("a", &cfg, |_, _| {}).unwrap();
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.token_count, 8);
    assert_eq!(a.tokens.len(), 8);
    assert!(!a.text.is_empty() || a.token_count > 0);
}

#[test]
fn eager_and_lazy_generation_match() {
    let path = tiny_model(5);
    let cfg = GenerateConfig {
        max_tokens: 6,
        temperature: 0.0,
        seed: 1,
        ..Default::default()
    };
    let opts_lazy = LoadOptions::default();
    let mut lazy = Engine::load(&path, opts_lazy).unwrap();
    let a = lazy.generate("hi", &cfg, |_, _| {}).unwrap();

    let opts_eager = LoadOptions {
        dequant: cortex::DequantPolicy::Eager,
        ..Default::default()
    };
    let mut eager = Engine::load(&path, opts_eager).unwrap();
    let b = eager.generate("hi", &cfg, |_, _| {}).unwrap();
    assert_eq!(a.tokens, b.tokens);
}

#[test]
fn kv_cache_grows_past_initial_capacity() {
    let path = tiny_model(6);
    let mut engine = Engine::load(&path, Default::default()).unwrap();
    let mut last = Vec::new();
    for pos in 0..24 {
        last = engine.model.forward(3, pos).unwrap();
    }
    assert!(last.iter().all(|v| v.is_finite()));
}

#[test]
fn out_of_range_token_is_rejected() {
    let path = tiny_model(7);
    let mut engine = Engine::load(&path, Default::default()).unwrap();
    assert!(matches!(
        engine.model.forward(1000, 0),
        Err(Error::InvalidToken(_))
    ));
    assert!(matches!(
        engine.model.forward(1, 64),
        Err(Error::ContextOverflow { .. })
    ));
}
