# Cortex — CUDA backend

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/status-working-brightgreen.svg)](#benchmarks)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## Status

**Implemented and working.** The CUDA backend accelerates the dominant cost of
inference — `matvec` — by keeping weights in VRAM and running a JIT-compiled
kernel. Attention, RMSNorm, RoPE, SwiGLU, sampling and tokenization still run on
CPU.

## Requirements

- NVIDIA GPU, compute capability ≥ 6.1 (tested on RTX 2060, SM 7.5)
- NVIDIA driver (`nvidia-smi`)
- CUDA Toolkit 12.x with NVRTC:
  - Ubuntu: `sudo apt-get install -y nvidia-cuda-toolkit`
  - Verify: `nvcc --version`
- VRAM: ~2 GB for a 0.5B model in f32 (test machine has 8 GB)

No Rust dependencies are added: the CUDA Driver API and NVRTC are loaded with
`dlopen` at runtime.

## Usage

```rust,no_run
use cortex::{Device, Engine, GenerateConfig, LoadOptions};

let options = LoadOptions {
    device: Device::Cuda(0),
    ..Default::default()
};
let mut engine = Engine::load("model.gguf", options)?;
let out = engine.generate("Hello", &GenerateConfig::default(), |_, s| print!("{s}"))?;
# Ok::<(), cortex::Error>(())
```

CLI:

```bash
cargo run --release --example generate -- model.gguf "Once upon a time" \
  --device cuda --temp 0 --max 64

cargo run --release --example bench -- model.gguf "Once upon a time" 64 --device cuda
cargo run --release --example cuda_bench   # kernel smoke test and GPU/CPU comparison
```

CUDA mode forces `DequantPolicy::Eager`: all matrices are dequantized to `f32`
once and uploaded to VRAM at load time. If CUDA is unavailable, use
`Device::Cpu` (the default).

## Implementation

- `src/cuda.rs` — FFI bindings for `cuInit`, `cuCtxCreate`, `cuModuleLoadData`,
  `cuLaunchKernel`, `cuMemcpyHtoD/DtoH`, plus NVRTC JIT compilation of a CUDA C
  kernel stored as a string.
- The kernel computes one output row per thread using vectorized `float4`
  loads and two independent accumulators.
- Reused scratch buffers avoid per-call allocations on `x`/`y`.
- Each layer uploads 7 matrices; the final projection is uploaded too. Small
  tensors (norms, embeddings) stay on CPU.

### Limitations

- Only `f32` weights in VRAM. A 7B model needs ~28 GB and will not fit into
  8 GB; it requires GPU kernels for Q4_K/Q8_0 or weight streaming.
- Single GPU (`Device::Cuda(0)`), no multi-GPU.
- No CUDA Graphs: each `matvec` issues a memcpy, a launch, a sync and a copy
  back. Overlap across layers is future work.
- Attention / KV cache are scalar on CPU.

## Benchmarks

`Qwen2.5-0.5B-Instruct-Q4_K_M`, one RTX 2060, 8 CPU threads for CPU paths,
30 generated tokens, prompt included:

| Configuration | Throughput |
|---|---|
| **Cortex CUDA** | **~36–38 tok/s** |
| llama.cpp b12755, CPU, 8 threads | ~24 tok/s |
| Cortex CPU, Lazy + SIMD | ~10 tok/s |

Model load (GGUF parsing + dequantization + ~2 GB upload) takes ~2.7 s and is
excluded from throughput.

The CUDA backend is therefore roughly **1.5× faster than llama.cpp** on this
machine, exceeding the initial +15–20% target.

## Roadmap

- [x] FFI wrapper for CUDA Driver API + NVRTC
- [x] Runtime JIT: CUDA C → PTX → `CUmodule`
- [x] `CudaDevice::upload` / `CudaDevice::matvec`
- [x] Integration into `Llama::forward` via `Device`
- [x] Benchmarks with `--device cuda`
- [ ] GPU kernels for Q4_K/Q8_0 (large models in VRAM)
- [ ] CUDA Graphs and stream overlap
- [ ] Attention / softmax on GPU

## Environment check

```bash
nvidia-smi
nvcc --version
ls /usr/local/cuda
```
