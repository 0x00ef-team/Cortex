# Benchmarks / Бенчмарки

> All numbers below were measured on the same machine. They are reproducible
> with the commands in this file. Measurements are noisy: the host had other
> workloads running (a Minecraft server and another Rust build), so treat small
> differences as insignificant.

## Environment / Окружение

| Component | Value |
|---|---|
| CPU | Intel Core i7-2600 @ 3.40 GHz, 4 cores / 8 threads |
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8 GB, driver 595.99.02 |
| CUDA Toolkit | 12.4 (NVRTC JIT; no `nvcc` at build time) |
| Rust | 1.100.0, `--release`, `lto=thin`, `codegen-units=1` |
| llama.cpp | b12755, CPU build and CUDA 12.8 build (`ngl=99`) |
| OS | Ubuntu, x86_64, AVX + SSE4.1 (no AVX2/FMA) |

Hardware has **AVX but not AVX2/FMA**, so Cortex uses SSE4.1 integer kernels
plus AVX float kernels with mul/add (no FMA).

---

## 1. Qwen2.5-0.5B-Instruct-Q4_K_M (494 M params)

File: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (~374 MiB), 30 generated tokens,
prompt included in Cortex throughput, 8 CPU threads where applicable.

| Configuration | Throughput | Command |
|---|---|---|
| **Cortex CUDA** | **~228 tok/s** | `examples/bench --device cuda` |
| llama.cpp CUDA, ngl=99 | **338 ± 37 tok/s** | `llama-bench -ngl 99 -n 32` |
| llama.cpp CUDA, prompt eval (pp4) | 795 ± 179 tok/s | same |
| llama.cpp CPU, 8 threads | ~24 tok/s | `llama-bench -n 8` |
| Cortex CPU, Lazy + SIMD + 8 threads | ~10 tok/s | `examples/bench --threads 8` |
| Cortex CPU, Eager + AVX | ~2–5 tok/s | `--eager` |

Cortex CUDA load time (GGUF parse + f32 dequant + ~2 GB upload): **~2.7 s**,
excluded from throughput.

**Conclusion:** Cortex CUDA beats llama.cpp **CPU** by ~1.7×, but is ~1.7× slower
than llama.cpp **CUDA**. The original "+15–20%" goal was against the CPU build.

### Reproduce

```bash
cargo build --release --examples

./target/release/examples/bench /tmp/qwen2-0.5b-q4km.gguf "Once upon a time" 30 --threads 8
./target/release/examples/bench /tmp/qwen2-0.5b-q4km.gguf "Once upon a time" 30 --device cuda

# llama.cpp (CUDA build from release b12755, runtime libs from cudart archive)
LD_LIBRARY_PATH=$PWD ./llama-bench -m /tmp/qwen2-0.5b-q4km.gguf -p 4 -n 32 -t 8 -ngl 99
```

---

## 2. Gemma 4 E4B IT Q4_K_XL (~4 B params)

File: `gemma-4-E4B-it-UD-Q4_K_XL.gguf` (4.8 GB). Prompt "Hello".

| Configuration | Throughput | Notes |
|---|---|---|
| Cortex CPU | ~0.27 tok/s | 12 tokens in 44.8 s; load 0.5 s with mmap |
| **Cortex CUDA** | **~4.1 tok/s** | 24 tokens in 5.9 s; upload 2 s; quantized kernels |
| llama.cpp CUDA, ngl=40 | ~3.1 tok/s | 15 tokens in 4.8 s |

Output cross-check (raw prompt `Hello`):

- llama.cpp: `! I're ready to chat. What's on your mind today?`
- Cortex CPU/CUDA: `! I'm ready to chat! What's on your mind today? 😊😄😃…`

Differences are attributed to quantization and accumulation order.

### Reproduce

```bash
cargo run --release --example gemma4 -- gemma4.gguf "Hello" --max 24
cargo run --release --example gemma4 -- gemma4.gguf "Hello" --max 24 --cuda

LD_LIBRARY_PATH=$PWD ./llama-completion -m gemma4.gguf -p "Hello" -n 16 -s 1 \
  --temp 0 -ngl 40 -no-cnv
```

---

## 3. Kernel microbenchmarks (`cargo bench`, 4096×4096, 1 thread)

| Kernel | Time | Effective throughput |
|---|---|---|
| F32 (AVX) | ~5.1 ms | ~6.5 GFLOP/s |
| Q5_0 fused SIMD | ~8–11 ms | ~3–4 GFLOP/s |
| Q4_K fused SIMD | ~6.3 ms | ~5.3 GFLOP/s |
| Q6_K fused SIMD (AVX) | ~7.7 ms | ~4.4 GFLOP/s |
| F16 lazy dequant | ~52 ms | ~0.6 GFLOP/s |

CUDA `matvec` (4096×4096 f32): **~0.67 ms** vs CPU ~5.2 ms.

---

## 4. Memory / Память

Gemma 4 (4.8 GB GGUF):

| Mode | Max RSS | Load time |
|---|---|---|
| Before mmap (copying tensors) | ~9 GB | 20–60 s |
| With mmap (current) | ~3.4 GB | 0.5 s CPU / 2 s CUDA |

---

## Caveats / Оговорки

- `llama-bench` `tg` excludes prompt evaluation; Cortex `bench` includes it.
- Host load varied between runs; numbers can move by 2–3×.
- Numbers are specific to this CPU/GPU and these quantizations.
- Cortex CUDA for LLaMA/Qwen2 stores f32 weights in VRAM (~2 GB for 0.5B);
  Gemma 4 keeps quantized weights and dequantizes on GPU.

---

## Русский

Все цифры замерены на одной машине: Intel i7-2600 (8 потоков), RTX 2060 SUPER
8 ГБ, Rust 1.100, llama.cpp b12755 (CPU и CUDA-сборки). CPU поддерживает AVX,
но **не AVX2/FMA**, поэтому в Cortex используются SSE4.1 + AVX без FMA.

### Qwen2.5-0.5B Q4_K_M

| Конфигурация | tok/s |
|---|---|
| Cortex CUDA | ~41–42 |
| llama.cpp CUDA (ngl=99) | ~338 |
| llama.cpp CPU (8 потоков) | ~24 |
| Cortex CPU (Lazy + SIMD) | ~10 |

Cortex CUDA быстрее llama.cpp **CPU** примерно в 1.7 раза, но медленнее
llama.cpp **CUDA** примерно в 8 раз. Изначальная цель "+15–20%" относилась
именно к CPU-сборке llama.cpp.

### Gemma 4 E4B Q4_K_XL (4.8 ГБ)

| Конфигурация | tok/s |
|---|---|
| Cortex CPU | ~0.27 |
| Cortex CUDA | ~4.1 |
| llama.cpp CUDA | ~3.1 |

Выходы обоих движков на промпт `Hello` совпадают по смыслу; различия — из-за
квантования и порядка накопления.

### Микро-benchmark (4096×4096)

F32 ~5.1 мс, Q4_K ~6.3 мс, Q6_K ~7.7 мс, Q5_0 ~8–11 мс, F16 lazy ~52 мс.
CUDA f32 matvec ~0.67 мс.

### Память

С mmap загрузка Gemma 4: 0.5 с и ~3.4 ГБ RSS вместо 20–60 с и ~9 ГБ.

### Важно

`llama-bench` исключает префилл, `examples/bench` включает. Нагрузка на хост
менялась между замерами, поэтому маленькие различия несущественны.

### CUDA Graph / MMQ / GPU argmax (Nov 26)

Qwen2.5-0.5B Q4_K_M, RTX 2060 SUPER, 40 tokens including prefill:

| Configuration | tok/s |
|---|---|
| CUDA, direct launches | 192–226 |
| CUDA, `CORTEX_CUDA_GRAPH=1` | 191–230 |
| CUDA, `CORTEX_MMQ=1` (Q4_K fused kernel) | ~190 |

Greedy CPU, CUDA and CUDA+graph produced byte-identical text for 30 tokens,
so all three paths are correctness-checked. Neither graph nor MMQ closed the
gap to llama.cpp CUDA (~338): the workload is dominated by weight streaming,
not launch overhead. GPU argmax removes the per-token 608 KB logits copy but
has little effect on total time.

CUDA Graph replay initially produced wrong logits for two reasons: all
`cuLaunchKernel` calls were pinned to the main stream while capture used
another stream (empty graph), and the layer loop was still executed on replay
(racing with the graph). Both fixed; capture mode is `0` and scalar kernel
arguments use stable heap slots.
