//! Autoregressive text generation.

use crate::error::Result;
use crate::model::Llama;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;

/// Parameters for a single generation run.
#[derive(Clone, Debug)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
    pub stop_on_eos: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 0,
            stop_on_eos: true,
        }
    }
}

/// Result of a generation run.
#[derive(Clone, Debug)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub token_count: usize,
}

/// Generate text from `prompt`.
///
/// `on_token` is invoked for every emitted token with its id and the newly
/// completed UTF-8 text fragment. It can be used to stream output to a UI.
pub fn generate<F>(
    model: &mut Llama,
    tokenizer: &Tokenizer,
    prompt: &str,
    cfg: &GenerateConfig,
    mut on_token: F,
) -> Result<GenerateOutput>
where
    F: FnMut(u32, &str),
{
    model.reset_cache();

    let mut prompt_ids = tokenizer.encode(prompt)?;
    if prompt_ids.is_empty() {
        prompt_ids.push(tokenizer.bos());
    }
    let ctx = model.context_length();
    if prompt_ids.len() > ctx {
        prompt_ids.truncate(ctx);
    }

    // Prefill: feed all prompt tokens through the transformer.
    let mut logits = Vec::new();
    for (pos, &id) in prompt_ids.iter().enumerate() {
        logits = model.forward(id, pos)?;
    }
    let mut pos = prompt_ids.len();

    let mut sampler = Sampler::new(cfg.seed);
    sampler.temperature = cfg.temperature;
    sampler.top_k = cfg.top_k;
    sampler.top_p = cfg.top_p;
    sampler.min_p = cfg.min_p;
    sampler.repeat_penalty = cfg.repeat_penalty;

    let mut all_ids = prompt_ids.clone();
    let mut produced: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut next = sampler.sample(&logits, &all_ids)?;

    for _ in 0..cfg.max_tokens {
        if cfg.stop_on_eos && next == tokenizer.eos() {
            break;
        }
        if pos >= ctx {
            break;
        }

        let bytes = tokenizer.decode_bytes(next)?;
        pending.extend_from_slice(&bytes);
        // Emit only complete UTF-8 codepoints; keep a partial tail for the
        // next token (tokens may split a multi-byte character).
        let fragment = match std::str::from_utf8(&pending) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                let n = e.valid_up_to();
                std::str::from_utf8(&pending[..n]).unwrap_or("").to_owned()
            }
        };
        if !fragment.is_empty() {
            let n = fragment.len();
            pending.drain(..n);
            text.push_str(&fragment);
            on_token(next, &fragment);
        }

        all_ids.push(next);
        produced.push(next);
        logits = model.forward(next, pos)?;
        pos += 1;
        next = sampler.sample(&logits, &all_ids)?;
    }

    if !pending.is_empty() {
        let tail = String::from_utf8_lossy(&pending).into_owned();
        text.push_str(&tail);
        on_token(next, &tail);
    }

    let token_count = produced.len();
    Ok(GenerateOutput {
        text,
        tokens: produced,
        token_count,
    })
}
