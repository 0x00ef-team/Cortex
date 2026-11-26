# Changelog / История изменений

## [0.1.0] — 2026-11-26

First public release. / Первый публичный релиз.

### Added / Добавлено

- GGUF v3 parser with zero-copy `mmap` loading / парсер GGUF v3 с zero-copy mmap
- Dequantization for F32/F16/BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0/Q8_1, Q2_K–Q6_K
- LLaMA/Mistral/Qwen2 transformer with GQA and growing KV cache
- Gemma 4 support: SWA/global layers, per-layer embeddings, shared KV, softcapping
- BPE tokenizer: SentencePiece, GPT-2/Qwen, Gemma 4 modes; byte fallback
- Sampling: greedy, temperature, top-k, top-p, min-p, repetition penalty
- AVX/SSE4.1 SIMD dequantization kernels and persistent thread pool
- CUDA backend: JIT kernels via NVRTC, f32 weights in VRAM for LLaMA/Qwen2
- CUDA quantized kernels (Q4_K/Q5_K/Q6_K/Q8_0/F16/F32) for Gemma 4
- Synthetic GGUF generator, 23 tests, benchmarks and 7 examples
- Bilingual README, BENCHMARKS.md, CONTRIBUTING.md

### Performance / Производительность

| Model / Модель | Cortex CPU | Cortex CUDA | llama.cpp CPU | llama.cpp CUDA |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | ~10 tok/s | ~42 tok/s | ~24 tok/s | ~338 tok/s |
| Gemma 4 E4B Q4_K_XL | ~0.27 tok/s | ~4.1 tok/s | — | ~3.1 tok/s |

See [BENCHMARKS.md](BENCHMARKS.md) for methodology and caveats.

[0.1.0]: https://github.com/0x00ef-team/cortex-llm/releases/tag/v0.1.0
