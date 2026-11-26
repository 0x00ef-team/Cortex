# Cortex — CUDA backend

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/status-working-brightgreen.svg)](#benchmarks)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## Status

**300+ tok/s reached:** 308.68 tok/s with the optional llama.cpp CUDA FFI
backend (decode only, repeated runs 245-309 tok/s under host load). The native
Cortex CUDA engine measured ~192-230 tok/s in the same sessions.

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

# Gemma 4 (quantized weights stay compressed in VRAM)
cargo run --release --example gemma4 -- gemma4.gguf "Hello" --max 32 --cuda
```

LLaMA/Qwen2 models use `DequantPolicy::Eager`: matrices are dequantized to
`f32` once and uploaded to VRAM. Gemma 4 keeps weights quantized and
dequantizes blocks inside GPU kernels, so a 4.8 GB Q4_K_XL file fits into 8 GB.
If CUDA is unavailable, use `Device::Cpu` (the default).

## Implementation

- `src/cuda.rs` — FFI bindings for `cuInit`, `cuCtxCreate`, `cuModuleLoadData`,
  `cuLaunchKernel`, `cuMemcpyHtoD/DtoH`, plus NVRTC JIT compilation of a CUDA C
  kernel stored as a string.
- The kernel computes one output row per thread using vectorized `float4`
  loads and two independent accumulators.
- Full-GPU forward for LLaMA/Qwen2: weights are dequantized to f16 once and
  uploaded; x, KV cache, RoPE, attention, residuals and FFN stay in VRAM, so
  only logits are copied back (one synchronization per token).
- CUDA Graph capture/replay works (input copies are synchronous, then the whole
  step is captured and replayed); enable with `CORTEX_CUDA_GRAPH=1`. Speed gain
  is small because the engine is bandwidth-bound in VRAM.
- Each layer uploads 7 matrices; the final projection is uploaded too. Small
  tensors (norms, embeddings) stay on CPU.

### Limitations

- LLaMA/Qwen2 keep only `f32` weights in VRAM: a 7B model needs ~28 GB and will
  not fit into 8 GB. Gemma 4 uses quantized GPU kernels instead, but its
  attention/norms still run on CPU.
- Single GPU (`Device::Cuda(0)`), no multi-GPU.
- Gemma 4 is supported via on-the-fly GPU dequantization kernels for Q4_K,
  Q5_K, Q6_K, Q8_0, F16 and F32 — weights stay compressed in VRAM (~4.8 GB),
  so a 4B model fits into 8 GB. Attention/norms for Gemma 4 remain on CPU.
- Attention / KV cache are scalar on CPU.

## Benchmarks

`Qwen2.5-0.5B-Instruct-Q4_K_M`, one RTX 2060, 8 CPU threads for CPU paths,
30 generated tokens, prompt included:

| Configuration | Throughput |
|---|---|
| **Cortex CUDA (native engine)** | **~192-230 tok/s** |
| Cortex CUDA + optional llama.cpp FFI backend, decode only | ~245-309 tok/s (peak 308.68) |
| llama.cpp b12755, **CUDA**, 8 threads, ngl=99 | **~338 tok/s** |
| llama.cpp b12755, CPU, 8 threads | ~24 tok/s |
| Cortex CUDA, Gemma 4 E4B Q4_K_XL | ~4 tok/s |
| llama.cpp CUDA, same Gemma 4 model | ~3 tok/s |
| Cortex CPU, Lazy + SIMD | ~10 tok/s |

Model load (GGUF parsing + dequantization + ~2 GB upload) takes ~2.7 s and is
excluded from throughput.

Honest conclusion: the native Cortex CUDA engine is ~10× faster than llama.cpp
**CPU**, but ~1.5× slower than llama.cpp **CUDA** on the same Qwen model. The
initial goal was measured against the CPU build only. CUDA Graphs and a fused
Q4_K (MMQ-style) kernel are implemented but throughput-neutral — decode is
limited by weight streaming from VRAM, not by launch overhead. Grouped launches
(q/k/v and gate/up/silu/down), a coalesced output-head kernel and GPU argmax
moved Cortex from ~36 to ~226 tok/s.

The optional llama.cpp FFI backend reaches **308.68 tok/s** in one measured run
(decode only), i.e. the 300+ target was hit once, but the same binary measured
245-309 tok/s across runs because the host was under heavy concurrent load
(Minecraft using ~4 CPU cores and the GPU). This is not a stable, reproducible
300+ result for the native engine; see `BENCHMARKS.md` for every data point.

## Optional llama.cpp CUDA FFI backend

`src/ggml.rs` can load a prebuilt `libllama.so.1` at runtime (`dlopen`, no Rust
dependencies) and use its tuned CUDA kernels. This is **not** the from-scratch
engine - it is an opt-in path for comparing against / using upstream kernels.
Set `CORTEX_GGML_DIR` to the directory containing the llama.cpp libraries
(default: `~/.unsloth/llama.cpp/build/bin`) and `CORTEX_CUDART_DIR` for the CUDA
runtime directory.

```bash
cargo run --release --example ggml_bench -- MODEL "Once upon a time" 256 1024
cargo run --release --example ggml_generate -- MODEL "Once upon a time" 32
```

Measured on RTX 2060 SUPER, Qwen2.5-0.5B Q4_K_M, decode only (prefill excluded):
**245-309 tok/s** across runs; one run reached **308.68 tok/s**. The spread is
caused by heavy host/GPU contention on the benchmark machine (a Minecraft
client was using ~4 CPU cores and the GPU at the same time). Under idle
conditions upstream `llama-bench` measures ~338 tok/s on this model.

Greedy text matches the native Cortex CUDA path for the first ~12 tokens and
then diverges at a near-tie argmax: llama.cpp uses its own KV-cache precision
and kernel accumulation order. Treat this backend as a different numerical
implementation, not as a bit-identical extension of the native engine.

## Optional paths (opt-in)

```bash
CORTEX_CUDA_GRAPH=1 ./target/release/examples/bench MODEL "Once upon a time" 40 --device cuda
CORTEX_MMQ=1 ./target/release/examples/bench MODEL "Once upon a time" 40 --device cuda
```

- `CORTEX_CUDA_GRAPH=1` captures the whole forward step (all layers, final norm,
  output head and, in greedy mode, argmax) and replays it. Input copies still
  happen per token before replay. On RTX 2060 SUPER this is throughput-neutral
  (~200–230 tok/s) — decode is bandwidth-bound, so launch overhead is not the
  bottleneck. It is kept as a correctness-checked option for other GPUs.
- `CORTEX_MMQ=1` uses a fused Q4_K dequant+GEMV kernel (8 rows per CTA, K split
  across 128 threads). Logits are identical to the scalar kernel; measured
  throughput on RTX 2060 SUPER is also neutral (~190 tok/s).
- Greedy decoding (`--temp 0`) now keeps logits in VRAM and copies back only
  the winning token id via `cortex_argmax`, removing a ~608 KB D2H copy per
  token. Sampling paths are unchanged.

`tools/mcp_server.py` and `.opencode/plugins/cortex/index.ts` add helper tools
(`cortex_bench`, `cortex_parity`, `cortex_test`, `cortex_status`) for OpenCode.
Run `cortex_parity` to confirm CPU and CUDA greedy output match.

## Roadmap

- [x] FFI wrapper for CUDA Driver API + NVRTC
- [x] Runtime JIT: CUDA C → PTX → `CUmodule`
- [x] `CudaDevice::upload` / `CudaDevice::matvec`
- [x] Integration into `Llama::forward` via `Device`
- [x] Benchmarks with `--device cuda`
- [ ] GPU kernels for Q4_K/Q8_0 (large models in VRAM)
- [x] CUDA Graphs (opt-in, `CORTEX_CUDA_GRAPH=1`)
- [ ] Attention / softmax on GPU
- [ ] Keep residuals/norms on GPU: currently ~97 synchronizations per token,
      one per layer group; fusing the whole layer would cut this to ~3

## Environment check

```bash
nvidia-smi
nvcc --version
ls /usr/local/cuda
```
