# Cortex — CUDA backend (roadmap)

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)](#status)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](README.md)

> Russian version below / Русская версия ниже.

---

## Status

**Not implemented yet.** This document describes the target design, prerequisites
and rollout plan for the CUDA backend of [Cortex](README.md). The CPU engine is
fully working; see the main README for its features and benchmarks.

## Why GPU

On CPU, Cortex evaluates `matvec`, attention and sampling on the processor. On
`Qwen2.5-0.5B Q4_K_M` with 8 threads it reaches ~4–6 tok/s versus ~24 tok/s for
llama.cpp. The dominant cost is `matvec`: every token reads all model weights
(~0.5 GB quantized, ~2 GB expanded to f32).

An RTX 2060 has roughly **300 GB/s** of VRAM bandwidth versus ~10 GB/s on CPU,
so moving `matvec` to the GPU should close most of the gap.

## Prerequisites

- NVIDIA GPU, compute capability ≥ 6.1 (test machine: RTX 2060, SM 7.5)
- NVIDIA driver — verify with `nvidia-smi`
- CUDA Toolkit 12.x with `nvcc` and NVRTC:
  - Ubuntu: `sudo apt-get install -y nvidia-cuda-toolkit`
  - Verify: `nvcc --version`
- VRAM: ~2 GB for a 0.5B model stored as f32; 8 GB on the test machine

## Target API

```rust,no_run
use cortex::{Engine, GenerateConfig, LoadOptions, Device};

let options = LoadOptions {
    device: Device::Cuda(0), // Device::Cpu remains the default
    ..Default::default()
};
let mut engine = Engine::load("model.gguf", options)?;
let out = engine.generate("Hello", &GenerateConfig::default(), |_, s| print!("{s}"))?;
# Ok::<(), cortex::Error>(())
```

If CUDA is unavailable, the plan is to fall back to CPU and emit a warning via
`LoadOptions::progress`.

## Design

### Moved to GPU

1. **Weights.** In CUDA mode all matrices are dequantized once to `f32`
   (`DequantPolicy::Eager`) and uploaded to VRAM at load time (~2 GB for 0.5B).
2. **`matvec` for large matrices** (`rows ≥ 256`): `out = W · x`. One CUDA block
   per group of rows, each thread computes one output element with a dot
   product over the row. Vectorized `float4` loads are planned.
3. **Final projection (logits)** uses the same kernel; results are copied back
   to host memory for sampling.

### Kept on CPU

- GGUF parsing, tokenizer, sampler, generation loop
- RMSNorm, RoPE, SwiGLU, attention and KV cache (phase 1)
- `cudaMemcpy` of `x`/`out` on each `matvec`

Rationale: attention operates on small matrices with a stateful KV cache, so its
GPU payoff is smaller and its implementation cost is higher. Phase 1 targets the
largest win with the least risk.

### Phase-1 limitations

- f32 weights only. A 7B model needs ~28 GB and will not fit into 8 GB; it
  requires GPU kernels for Q4_K/Q8_0 or weight streaming.
- Single GPU (`Device::Cuda(0)`), no multi-GPU.
- No CUDA Graphs: ~170 launches/token × ~5–10 µs ≈ 1–2 ms/token overhead.

## Build (planned)

```bash
cargo build --release --features cuda   # with CUDA Toolkit in PATH
cargo build --release                   # plain CPU build, zero dependencies
```

A `cuda` feature is planned in `Cargo.toml`. The backend will use the CUDA
Driver API through FFI (`cuInit`, `cuDeviceGet`, `cuCtxCreate`,
`cuModuleLoadData`, `cuLaunchKernel`, `cuMemcpyHtoD` / `cuMemcpyDtoH`). Kernel
sources will be stored as CUDA C strings and JIT-compiled to PTX with NVRTC at
runtime, so end users do not need a separate `nvcc` step.

## Expected performance

| Configuration | Expected speed |
|---|---|
| Cortex CPU, Lazy + SIMD | ~4–6 tok/s |
| Cortex CPU, Eager + AVX | ~2–5 tok/s |
| llama.cpp CPU (reference) | ~24 tok/s |
| Cortex CUDA, f32 `matvec` | ~30–80 tok/s (launch-overhead dependent) |

## Roadmap

- [ ] FFI wrapper for CUDA Driver API + NVRTC
- [ ] Runtime JIT: CUDA C → PTX → `CUmodule`
- [ ] `CudaDevice::upload` / `CudaDevice::matvec`
- [ ] Wire into `Llama::forward` via `Device`
- [ ] `examples/bench.rs --device cuda`
- [ ] GPU kernels for Q4_K/Q8_0 (large models)
- [ ] CUDA Graphs to reduce launch overhead

## Environment check

```bash
nvidia-smi         # GPU and driver
nvcc --version     # Toolkit
ls /usr/local/cuda # Install path
```

---

## Русская версия

### Статус

**Пока не реализовано.** Здесь описан целевой дизайн CUDA-бэкенда для
[Cortex](README.md). CPU-движок полностью рабочий.

### Зачем GPU

Основная стоимость генерации — `matvec`: на каждом токене считываются все
веса модели. У RTX 2060 пропускная способность VRAM ~300 ГБ/с против ~10 ГБ/с
у CPU, поэтому перенос `matvec` на GPU должен почти полностью закрыть разрыв с
llama.cpp (~24 ток/с).

### Требования

- NVIDIA GPU, compute capability ≥ 6.1 (тестовая машина — RTX 2060, SM 7.5)
- Драйвер NVIDIA (`nvidia-smi`)
- CUDA Toolkit 12.x с `nvcc` и NVRTC
- ~2 ГБ VRAM для модели 0.5B в f32 (на тестовой машине 8 ГБ)

### Дизайн

На GPU: веса в `f32` (загружаются один раз), `matvec` для больших матриц и
последний слой с logits. На CPU: GGUF, токенизатор, сэмплинг, RMSNorm, RoPE,
SwiGLU, attention и KV-кэш (первый этап).

Ограничения первого этапа: только f32-веса (7B не поместится в 8 ГБ), один
GPU, без CUDA Graphs.

### План

- [ ] FFI к CUDA Driver API + NVRTC
- [ ] JIT-компиляция CUDA C → PTX в рантайме
- [ ] `CudaDevice::upload` / `CudaDevice::matvec`
- [ ] Интеграция в `Llama::forward` через `Device`
- [ ] Бенчмарк с `--device cuda`
- [ ] Ядра Q4_K/Q8_0 для больших моделей
- [ ] CUDA Graphs
