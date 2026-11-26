# Cortex

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#build)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](#tests-without-a-model)
[![CUDA](https://img.shields.io/badge/CUDA-working-brightgreen.svg)](README_CUDA.md)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

---

## English

**Cortex** is an LLM inference engine written from scratch in Rust, with **zero
external dependencies** for the CPU path — no `ndarray`, no `candle`, no
`llama.cpp`, no tokenizer crates. Just `std`.

It runs LLaMA-family GGUF models (LLaMA 2/3, Mistral, Qwen2, CodeLlama, etc.)
and **Gemma 4** dense text models, including quantized weights, and offers an
optional CUDA backend that is ~1.7× faster than llama.cpp **CPU** on the test
machine (but ~8× slower than llama.cpp CUDA). GGUF files are memory-mapped, so
loading a 4.8 GB model takes ~0.5 s.

### Features

| Subsystem | What is implemented |
|---|---|
| GGUF v3 | Header, metadata and tensor descriptors; tensor data is memory-mapped (zero-copy) |
| Architectures | LLaMA/Mistral/Qwen2-style transformers plus Gemma 4 (SWA/global layers, per-layer embeddings, shared KV, softcapping) |
| Quantization | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K–Q6_K — all dequantizers ported from the reference ggml C code |
| Math | Unrolled `dot`, AVX/SSE4.1 SIMD fused kernels, RMSNorm, stable softmax, RoPE (LLaMA + NeoX), SwiGLU |
| Transformer | LLaMA-style blocks, grouped-query attention, geometrically growing KV cache, Qwen attention biases |
| Tokenizer | BPE built from GGUF vocabularies, SentencePiece and GPT-2/Qwen modes, explicit merge list, `<0xXX>` byte fallback |
| Sampling | greedy, temperature, top-k, top-p, min-p, repetition penalty, deterministic xorshift64 RNG |
| Generation | Prefill, streaming UTF-8-safe output, stop on EOS |
| CUDA | JIT-compiled `matvec`, weights in VRAM, grouped launches, GPU SwiGLU |

### Build

```bash
cd cortex
cargo build --release
cargo test
cargo bench
```

### Usage

```rust,no_run
use cortex::{Engine, GenerateConfig, LoadOptions};

let mut engine = Engine::load("model.gguf", LoadOptions::default())?;
let cfg = GenerateConfig {
    max_tokens: 64,
    temperature: 0.8,
    top_k: 40,
    top_p: 0.9,
    ..Default::default()
};
let out = engine.generate("Hello", &cfg, |_id, piece| print!("{piece}"))?;
println!("\n{} tokens", out.token_count);
# Ok::<(), cortex::Error>(())
```

CLI:

```bash
cargo run --release --example generate -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf "Once upon a time" \
  --temp 0.8 --top-p 0.9 --max 128 --seed 42 --threads 8

cargo run --release --example info -- model.gguf
cargo run --release --example tokens -- model.gguf "Hello world"
cargo run --release --example bench -- model.gguf "Once upon a time" 64 --threads 8

# Gemma 4 (CPU)
cargo run --release --example gemma4 -- gemma4.gguf "Hello" --max 32
```

GPU:

```bash
cargo run --release --example generate -- model.gguf "Hello" --device cuda
cargo run --release --example bench -- model.gguf "Hello" 64 --device cuda
```

See [README_CUDA.md](README_CUDA.md) for the CUDA backend design and benchmarks.

### Performance

`Qwen2.5-0.5B-Instruct-Q4_K_M`, RTX 2060, 8 CPU threads where applicable:

| Configuration | Throughput |
|---|---|
| **Cortex CUDA** | **~41–42 tok/s** |
| llama.cpp b12755, CUDA, 8 threads | ~338 tok/s |
| llama.cpp b12755, CPU, 8 threads | ~24 tok/s |
| Cortex CPU, Lazy + SIMD | ~4–10 tok/s |

The original goal of beating llama.cpp **CPU** by 15–20% is exceeded (~1.7×),
but llama.cpp CUDA is still ~8× faster. CPU optimizations include AVX `dot`, fused Q8_0/Q5_0/Q4_K/Q6_K
kernels, block-wise dequantization into an L1-resident buffer, and a persistent
thread pool.

### Tests without a model

The `cortex::synthetic` module generates a tiny valid GGUF (2 layers, dim 16,
vocab 32), so integration tests exercise the full pipeline without downloading
anything:

```bash
cargo test
cargo run --release --example make_tiny -- /tmp/tiny.gguf
```

Source layout:

```
src/
  quant/dequant.rs   dequantization kernels for all supported GGML types
  gguf/              GGUF container parser
  tensor.rs          Matrix, matvec, lazy/eager dequantization
  tokenizer/bpe.rs   BPE and text normalization
  model/             config, attention kernels, KV cache, weights, forward pass
  cuda.rs            CUDA driver FFI, NVRTC JIT, GPU matvec
  model/gemma4.rs    Gemma 4 architecture (SWA, per-layer embeddings, shared KV)
  sampler.rs         next-token sampling
  generate.rs        generation loop
  synthetic.rs       test-model generator
```

### Limitations

- GGUF version 3 only.
- 1-D and 2-D tensors; no MoE support.
- CPU path is scalar/SIMD Rust: no GPU required, but slower than native C++.
- CUDA path stores f32 weights in VRAM: a 7B model needs ~28 GB.
- Exotic quantizations (IQ-*, TQ*) are not implemented.
- Gemma 4 CUDA uses on-the-fly dequantization kernels and keeps weights quantized in VRAM (~4.8 GB).

### License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

---

## Русский

**Cortex** — LLM inference-движок, написанный с нуля на Rust. CPU-путь не имеет
внешних зависимостей: никаких `ndarray`, `candle`, `llama.cpp` или crates для
токенизации — только `std`.

Движок запускает GGUF-модели семейства LLaMA (LLaMA 2/3, Mistral, Qwen2,
CodeLlama и др.) и плотные текстовые модели **Gemma 4**, включая квантованные
веса. GGUF-файлы отображаются в память через mmap, поэтому загрузка модели
на 4.8 ГБ занимает ~0.5 с. Есть опциональный CUDA-бэкенд, который на тестовой
машине **примерно в 1.7 раза быстрее llama.cpp CPU** (но ~в 8 раз медленнее llama.cpp CUDA).

### Возможности

| Подсистема | Что реализовано |
|---|---|
| GGUF v3 | Парсер заголовка и метаданных; данные memory-mapped (без копирования) |
| Архитектуры | Трансформеры LLaMA/Mistral/Qwen2 и Gemma 4 (SWA/global слои, per-layer embeddings, общий KV, softcapping) |
| Квантование | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K–Q6_K — деквантовщики портированы с эталонного C-кода ggml |
| Математика | Развёрнутый `dot`, AVX/SSE4.1 SIMD-ядра, RMSNorm, стабильный softmax, RoPE (LLaMA и NeoX), SwiGLU |
| Трансформер | Блоки LLaMA, grouped-query attention, KV-кэш с геометрическим ростом, attention-биасы Qwen |
| Токенизатор | BPE из GGUF-словаря, режимы SentencePiece и GPT-2/Qwen, явный список `merges`, byte-fallback `<0xXX>` |
| Сэмплинг | greedy, temperature, top-k, top-p, min-p, repetition penalty, детерминированный xorshift64 |
| Генерация | Префилл, потоковая выдача с корректным UTF-8, stop by EOS |
| CUDA | JIT-компилируемый `matvec`, веса в VRAM, групповые запуски, SwiGLU на GPU |

### Сборка

```bash
cd cortex
cargo build --release
cargo test
cargo bench
```

### Использование

```rust,no_run
use cortex::{Engine, GenerateConfig, LoadOptions};

let mut engine = Engine::load("model.gguf", LoadOptions::default())?;
let cfg = GenerateConfig {
    max_tokens: 64,
    temperature: 0.8,
    top_k: 40,
    top_p: 0.9,
    ..Default::default()
};
let out = engine.generate("Hello", &cfg, |_id, piece| print!("{piece}"))?;
println!("\n{} токенов", out.token_count);
# Ok::<(), cortex::Error>(())
```

CLI:

```bash
cargo run --release --example generate -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf "Once upon a time" \
  --temp 0.8 --top-p 0.9 --max 128 --seed 42 --threads 8

cargo run --release --example info -- model.gguf
cargo run --release --example tokens -- model.gguf "Hello world"
cargo run --release --example bench -- model.gguf "Once upon a time" 64 --threads 8

# Gemma 4 (CPU)
cargo run --release --example gemma4 -- gemma4.gguf "Hello" --max 32
```

GPU:

```bash
cargo run --release --example generate -- model.gguf "Hello" --device cuda
cargo run --release --example bench -- model.gguf "Hello" 64 --device cuda
```

Дизайн и замеры CUDA-бэкенда: [README_CUDA.md](README_CUDA.md).

### Производительность

`Qwen2.5-0.5B-Instruct-Q4_K_M`, RTX 2060, 8 потоков где применимо:

| Конфигурация | Скорость |
|---|---|
| **Cortex CUDA** | **~41–42 ток/с** |
| llama.cpp b12755, CUDA, 8 потоков | ~338 ток/с |
| llama.cpp b12755, CPU, 8 потоков | ~24 ток/с |
| Cortex CPU, Lazy + SIMD | ~4–10 ток/с |

Изначальная цель — обогнать llama.cpp **CPU** на 15–20% — превышена с CUDA-бэкендом
(~1.7×), но llama.cpp CUDA всё ещё ~в 8 раз быстрее. CPU-оптимизации: AVX-векторизация `dot`, SIMD-ядра для Q8_0/Q5_0/Q4_K/Q6_K,
деквантовка блоками в L1-буфер и постоянный пул потоков.

### Тесты без модели

Модуль `cortex::synthetic` генерирует крошечный валидный GGUF (2 слоя, dim 16,
vocab 32), поэтому интеграционные тесты прогоняют весь конвейер без загрузки модели:

```bash
cargo test
cargo run --release --example make_tiny -- /tmp/tiny.gguf
```

Структура исходников:

```
src/
  quant/dequant.rs   деквантовка всех поддерживаемых GGML-типов
  gguf/              парсер контейнера GGUF
  tensor.rs          Matrix, matvec, ленивая/жадная деквантовка
  tokenizer/bpe.rs   BPE и нормализация текста
  model/             конфиг, ядра attention, KV-кэш, веса, forward pass
  cuda.rs            FFI к CUDA Driver API, NVRTC JIT, GPU matvec
  model/gemma4.rs    архитектура Gemma 4 (SWA, per-layer embeddings, общий KV)
  sampler.rs         выбор следующего токена
  generate.rs        цикл генерации
  synthetic.rs       генератор тестовой модели
```

### Ограничения

- Только GGUF версии 3.
- Одномерные и двумерные тензоры; MoE не поддерживается.
- CPU-путь — скалярный/SIMD Rust: GPU не нужен, но скорость ниже, чем у C++.
- CUDA-путь хранит f32-веса в VRAM: модель 7B требует ~28 ГБ.
- Экзотические кванты (IQ-*, TQ*) не реализованы.
- CUDA-путь Gemma 4 деквантует веса прямо на GPU и держит их в VRAM в квантованном виде (~4.8 ГБ).

### Лицензия

Двойная лицензия: [MIT](LICENSE-MIT) или [Apache-2.0](LICENSE-APACHE) на выбор.
