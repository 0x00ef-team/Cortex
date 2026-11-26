# Contributing / Участие в разработке

Thanks for your interest in Cortex! This document explains how to build the
project, where to put new code and what a good pull request looks like.

## Quick start

```bash
git clone https://github.com/0x00ef-team/Cortex.git
cd Cortex
cargo build --release
cargo test
cargo run --release --example make_tiny -- /tmp/tiny.gguf
cargo run --release --example generate -- /tmp/tiny.gguf "Hello" --max 32
```

Requirements: Rust 1.80+. CUDA is optional (Driver + NVRTC at runtime only).

## Where things live

| Area | File |
|---|---|
| GGUF container / mmap | `src/gguf/` |
| Dequantization kernels | `src/quant/dequant.rs` |
| Tensor storage and `matvec` | `src/tensor.rs` |
| LLaMA/Qwen2 model | `src/model/{config,llama,weights,cache,attention}.rs` |
| Gemma 4 model | `src/model/gemma4.rs` |
| BPE tokenizer | `src/tokenizer/bpe.rs` |
| Sampling / generation | `src/sampler.rs`, `src/generate.rs` |
| CUDA backend and JIT kernels | `src/cuda.rs` |
| Synthetic test model | `src/synthetic.rs` |

## Rules

1. **No external dependencies** in the CPU path. If you need a crate, open an
   issue first.
2. Every new dequantization kernel must have a unit test comparing it against
   the scalar reference implementation (`dequantize_row`).
3. Every new architecture needs an end-to-end test using a generated tiny GGUF
   where possible.
4. Never break `cargo test`. Run `cargo build --release --examples` too.
5. Keep claims in READMEs honest and backed by `BENCHMARKS.md`.

## Commit messages

Short imperative summary, e.g.:

```
Add SIMD dequant kernel for Q3_K
Fix Q8_1 field-order detection on legacy GGUF files
Document CUDA benchmark methodology
```

## Pull requests

- One logical change per PR.
- Describe what changed, how you tested it and what you measured.
- For performance changes, include before/after numbers from `cargo bench` or
  `examples/bench`.
- Link the relevant issue if there is one.

## Good first issues

Look for issues labeled `good first issue` and `help wanted`. Typical tasks:

- SIMD dequant kernels for Q2_K / Q3_K (port `dequant_q2_k` / `dequant_q3_k`);
- CI job that runs benchmarks and publishes results;
- unit tests for malformed GGUF metadata;
- documentation and examples.

## License

By contributing you agree that your contribution is dual-licensed under MIT
or Apache-2.0, at the project's option (see `LICENSE`).

---

## Русский

Спасибо за интерес к Cortex!

### Быстрый старт

```bash
git clone https://github.com/0x00ef-team/Cortex.git
cd Cortex
cargo build --release
cargo test
cargo run --release --example make_tiny -- /tmp/tiny.gguf
cargo run --release --example generate -- /tmp/tiny.gguf "Hello" --max 32
```

Нужен Rust 1.80+. CUDA необязательна: драйвер и NVRTC подгружаются в рантайме.

### Правила

1. CPU-путь остаётся **без внешних зависимостей**. Нужна новая зависимость — сначала issue.
2. Любое новое ядро деквантовки обязано иметь тест на совпадение со скалярной реализацией.
3. Новая архитектура — по возможности с end-to-end тестом на крошечном GGUF.
4. `cargo test` всегда зелёный; дополнительно проверяй `cargo build --release --examples`.
5. Цифры в README должны подтверждаться `BENCHMARKS.md`.

### Pull request

Одно логическое изменение на PR. Опиши, что изменилось, как проверял и что
замерил. Для performance-правок приложи before/after из `cargo bench` или
`examples/bench`.

### Хорошие первые задачи

Иши issue с метками `good first issue` и `help wanted`: SIMD-ядра Q2_K/Q3_K,
CI с бенчмарками, тесты битых GGUF-метаданных, документация.

Лицензия: двойная MIT/Apache-2.0 (см. `LICENSE`).
