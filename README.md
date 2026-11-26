# Cortex

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](#сборка)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](#тесты-без-модели)
[![CUDA](https://img.shields.io/badge/CUDA-roadmap-yellow.svg)](README_CUDA.md)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#лицензия)

**Cortex** — LLM inference-движок, написанный с нуля на Rust. Без внешних зависимостей: никаких `ndarray`, ` candle`, `llama.cpp` или tokenizers. Только `std`.

Поддерживаются обычные GGUF-модели семейства LLaMA (LLaMA 2/3, Mistral, Qwen2, CodeLlama и т.п.) с квантованными весами.

## Что реализовано

| Подсистема | Что внутри |
|---|---|
| GGUF v3 | парсер заголовка, метаданных и дескрипторов тензоров; ленивое чтение данных напрямую из файла |
| Квантование | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K — все деквантовщики портированы с эталонных C-реализаций ggml |
| Математика | `matvec` с развёрнутыми аккумуляторами, RMSNorm, softmax, RoPE (LLaMA и NeoX), SwiGLU |
| Трансформер | LLaMA-архитектура, grouped-query attention (GQA), KV-кэш с геометрическим ростом |
| Токенизатор | BPE по GGUF-словарю, режимы SentencePiece (`▁`) и GPT-2/Qwen (byte→unicode), byte-fallback `<0xXX>` |
| Сэмплинг | greedy, temperature, top-k, top-p (nucleus), min-p, repetition penalty, детерминированный xorshift64 RNG |
| Генерация | префилл, потоковая выдача по завершённым UTF-8- кодпоинтам, stop-by-EOS |

## Сборка

```bash
cd cortex
cargo build --release
cargo test
cargo bench
```

## Использование

### CLI

```bash
# Скачать, например, Q4_K_M:
# https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

cargo run --release --example generate -- \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "Once upon a time" \
  --temp 0.8 --top-p 0.9 --max 128 --seed 42
```

Флаги: `--temp`, `--max`, `--top-k`, `--top-p`, `--seed`, `--eager`.

### Библиотека

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

Ленивая деквантовка (по умолчанию) держат веса в компактном виде и деквантуют по строке на каждый `matvec`. Если памяти достаточно и нужна максимум скорости — `LoadOptions { dequant: DequantPolicy::Eager, .. }`.

### Инспекция GGUF

```bash
cargo run --release --example info -- model.gguf
```

## Производительность (честно)

На `Qwen2.5-0.5B-Instruct-Q4_K_M`, 8 потоков, один и тот же CPU:

| Движок | Декодирование |
|---|---|
| **Cortex CUDA** (RTX 2060) | **~41–42 ток/с** |
| llama.cpp b12755 (8 threads) | ~24 ток/с |
| Cortex, `Lazy` + SIMD (AVX/SSE4.1) + 8 потоков | ~4–10 ток/с |

Цель «на 15–20% быстрее llama.cpp» **достигнута с CUDA-бэкендом** (~1.7×). CPU-путь остаётся dependency-free и уступает. Что сделано:

- AVX-векторизация `dot` (≈3× против скалярного цикла);
- SIMD-ядра `dot_quant` для **Q8_0, Q5_0, Q4_K и Q6_K** (8-полосные AVX + SSE4.1-fallback); таблица расширения битов для Q5_0 убрала скалярный цикл;
- деквантовка блоками в L1-буфер вместо материализации всей строки;
- **постоянный пул потоков** с барьером — потоки создаются один раз при загрузке модели, а не на каждый `matvec`;
- режим `Eager` для скорости ценой памяти.

Дальнейший резерв — SIMD-ядра для Q2_K/Q3_K, векторизация attention и prefetch весов.

На машине с AVX2/FMA запуск со специфической сборкой даёт дополнительный выигрыш:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

```bash
cargo run --release --example bench -- model.gguf "Once upon a time" 16 --threads 8
cargo bench
```

## GPU

CUDA-бэкенд **реализован**: `matvec` выполняется на GPU, ~36 ток/с на Qwen2.5-0.5B Q4_K_M (против ~24 у llama.cpp). Подробности: [README_CUDA.md](README_CUDA.md).

## Тесты без модели

Модуль `cortex::synthetic` умеет генерировать крошечный валидный GGUF (2 слоя, dim 16, vocab 32). На нём интеграционные тесты прогоняют весь конвейер: GGUF → токенизатор → трансформер → сэмплинг.

```bash
cargo test
```

## Архитектура исходников

```
src/
  quant/dequant.rs   деквантовка всех поддерживаемых GGML-типов
  gguf/              бинарный формат контейнера
  tensor.rs          Matrix, matvec, ленивая/жадная деквантовка
  tokenizer/bpe.rs   BPE и нормализация текста
  model/
    config.rs        гиперпараметры из метаданных
    attention.rs     RMSNorm, RoPE, softmax, SwiGLU
    cache.rs         KV-кэш
    weights.rs       загрузка тензоров
    llama.rs         прямой проход transformer'а
  sampler.rs         выбор следующего токена
  generate.rs        цикл генерации
  synthetic.rs       генератор тестовой модели
```

## Ограничения

- GGUF версии 3 только (актуальная версия llama.cpp).
- Поддерживаются 1- и 2-мерные тензоры; MoE-архитектуры не реализованы.
- Нет GPU, SIMD и многопоточности — чистый скалярный Rust. Это учебный и понятный код, а не рекордная скорость.
- Некоторые экзотические кванты (IQ-*, TQ*) пока не поддерживаются.

## Лицензия

MIT OR Apache-2.0 на выбор.
