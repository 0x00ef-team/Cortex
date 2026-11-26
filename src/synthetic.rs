//! Writer for a tiny synthetic GGUF model.
//!
//! This exists so the whole pipeline (GGUF -> tokenizer -> transformer ->
//! sampling) can be exercised in tests without downloading a real model.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::Result;
use crate::quant::ggml_type;

const N_LAYERS: usize = 2;
const N_EMBD: usize = 16;
const N_FFN: usize = 32;
const N_HEADS: usize = 4;
const N_KV: usize = 2;
const CTX: usize = 32;
const VOCAB: usize = 32;

/// Write a deterministic, randomly initialized 2-layer GGUF model.
pub fn write_tiny_model(path: impl AsRef<Path>, seed: u64) -> Result<()> {
    let mut w = Writer::new();
    w.bytes(b"GGUF");
    w.u32(3); // version

    let tensor_names = tensor_names();
    w.u64(tensor_names.len() as u64);
    w.u64(20); // metadata key/value count

    // ---- metadata ----
    w.str_key("general.architecture", b"llama");
    w.str_key("general.name", b"cortex-tiny");
    w.u32_key("llama.block_count", N_LAYERS as u32);
    w.u32_key("llama.embedding_length", N_EMBD as u32);
    w.u32_key("llama.feed_forward_length", N_FFN as u32);
    w.u32_key("llama.attention.head_count", N_HEADS as u32);
    w.u32_key("llama.attention.head_count_kv", N_KV as u32);
    w.f32_key("llama.attention.layer_norm_rms_epsilon", 1e-5);
    w.u32_key("llama.context_length", CTX as u32);
    w.f32_key("llama.rope.freq_base", 10_000.0);
    w.u32_key("llama.vocab_size", VOCAB as u32);

    w.key("tokenizer.ggml.model");
    w.u32(8); // STRING
    w.str_bytes(b"llama");

    w.key("tokenizer.ggml.tokens");
    w.u32(9); // ARRAY
    w.u32(8); // of STRING
    w.u64(VOCAB as u64);
    for t in vocab_tokens() {
        w.str_bytes(&t);
    }

    w.key("tokenizer.ggml.scores");
    w.u32(9);
    w.u32(6); // F32
    w.u64(VOCAB as u64);
    for i in 0..VOCAB {
        w.f32(i as f32 * 0.01);
    }

    w.key("tokenizer.ggml.token_type");
    w.u32(9);
    w.u32(4); // U32
    w.u64(VOCAB as u64);
    for _ in 0..VOCAB {
        w.u32(1);
    }

    w.u32_key("tokenizer.ggml.bos_token_id", 1);
    w.u32_key("tokenizer.ggml.eos_token_id", 2);
    w.u32_key("tokenizer.ggml.unknown_token_id", 0);
    w.bool_key("tokenizer.ggml.add_bos_token", true);
    w.bool_key("tokenizer.ggml.add_eos_token", false);

    // ---- tensor descriptors ----
    // Offsets are relative to the start of the aligned data section.
    let mut offsets = Vec::new();
    // Alignment is computed from the header size, which we know only after
    // writing the descriptors, so use a fixed 32-byte alignment and pad the
    // header with a dummy `general.alignment` value of 32 (the default).
    let mut running = 0usize;
    for (name, dims) in &tensor_names {
        let elements: usize = dims.iter().product();
        offsets.push(running);
        w.str_bytes(name.as_bytes());
        w.u32(dims.len() as u32);
        for &d in dims {
            w.u64(d as u64);
        }
        w.u32(ggml_type::F32);
        w.u64(running as u64);
        running += elements * 4;
    }

    // ---- tensor data ----
    let pad = (32 - w.len() % 32) % 32;
    w.zeros(pad);
    let data_start = w.len();
    debug_assert_eq!(data_start % 32, 0);

    let mut rng = Lcg::new(seed);
    for (name, dims) in &tensor_names {
        let elements: usize = dims.iter().product();
        if name.contains("norm") {
            for _ in 0..elements {
                w.f32(1.0);
            }
        } else {
            for _ in 0..elements {
                w.f32((rng.next_f64() as f32 * 2.0 - 1.0) * 0.05);
            }
        }
    }

    let file = File::create(path.as_ref())?;
    let mut out = BufWriter::new(file);
    out.write_all(&w.buf)?;
    out.flush()?;
    Ok(())
}

fn vocab_tokens() -> Vec<Vec<u8>> {
    let mut v = vec![b"<unk>".to_vec(), b"<s>".to_vec(), b"</s>".to_vec()];
    for c in b'a'..=b'z' {
        v.push(vec![c]);
    }
    for c in [b' ', b'.', b',', b'!'] {
        v.push(vec![c]);
    }
    v.resize(VOCAB, b"?".to_vec());
    v
}

fn tensor_names() -> Vec<(String, Vec<usize>)> {
    let mut t = vec![("token_embd.weight".to_string(), vec![N_EMBD, VOCAB])];
    for l in 0..N_LAYERS {
        let p = |s: &str| (format!("blk.{l}.{s}"), vec![]);
        t.push(p("attn_norm.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD];
        t.push(p("attn_q.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_HEADS * (N_EMBD / N_HEADS)];
        t.push(p("attn_k.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_KV * (N_EMBD / N_HEADS)];
        t.push(p("attn_v.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_KV * (N_EMBD / N_HEADS)];
        t.push(p("attn_output.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_EMBD];
        t.push(p("ffn_norm.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD];
        t.push(p("ffn_gate.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_FFN];
        t.push(p("ffn_up.weight"));
        t.last_mut().unwrap().1 = vec![N_EMBD, N_FFN];
        t.push(p("ffn_down.weight"));
        t.last_mut().unwrap().1 = vec![N_FFN, N_EMBD];
    }
    t.push(("output_norm.weight".to_string(), vec![N_EMBD]));
    t.push(("output.weight".to_string(), vec![N_EMBD, VOCAB]));
    t
}

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn len(&self) -> usize {
        self.buf.len()
    }
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str_bytes(&mut self, s: &[u8]) {
        self.u64(s.len() as u64);
        self.bytes(s);
    }
    fn key(&mut self, s: &str) {
        self.str_bytes(s.as_bytes());
    }
    fn str_key(&mut self, k: &str, v: &[u8]) {
        self.key(k);
        self.u32(8);
        self.str_bytes(v);
    }
    fn u32_key(&mut self, k: &str, v: u32) {
        self.key(k);
        self.u32(4);
        self.u32(v);
    }
    fn f32_key(&mut self, k: &str, v: f32) {
        self.key(k);
        self.u32(6);
        self.f32(v);
    }
    fn bool_key(&mut self, k: &str, v: bool) {
        self.key(k);
        self.u32(7);
        self.buf.push(v as u8);
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
