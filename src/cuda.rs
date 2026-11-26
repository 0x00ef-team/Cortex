//! Minimal CUDA driver backend: dynamic loading, NVRTC JIT and `matvec`.
//!
//! No Rust crate dependencies — the CUDA Driver API and NVRTC are loaded at
//! runtime via `dlopen`. If the libraries are missing, `CudaDevice::new`
//! returns an error and the CPU backend can be used instead.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::error::{Error, Result};

const CUDA_SUCCESS: i32 = 0;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

macro_rules! ffi {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let sym = unsafe { dlsym($lib, concat!($name, "\0").as_ptr() as *const c_char) };
        if sym.is_null() {
            return Err(Error::InvalidConfig(format!("CUDA symbol {} not found", $name)));
        }
        std::mem::transmute::<*mut c_void, $ty>(sym)
    }};
}

type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> i32;
type CuCtxCreate = unsafe extern "C" fn(*mut u64, u32, c_int) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut u64, *const c_void) -> i32;
type CuModuleGetFunction = unsafe extern "C" fn(*mut u64, u64, *const c_char) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    u64, u32, u32, u32, u32, u32, u32, u32, u64, *mut *mut c_void, *mut c_void,
) -> i32;
type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
type CuStreamCreate = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuStreamBeginCapture = unsafe extern "C" fn(u64, u32) -> i32;
type CuStreamEndCapture = unsafe extern "C" fn(u64, *mut u64) -> i32;
type CuGraphInstantiate = unsafe extern "C" fn(*mut u64, u64, *mut u64, *mut c_char, usize, u32) -> i32;
type CuGraphLaunch = unsafe extern "C" fn(u64, u64) -> i32;
type CuStreamSynchronize = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, u64) -> i32;

type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut u64,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> i32;
type NvrtcCompileProgram = unsafe extern "C" fn(u64, c_int, *const *const c_char) -> i32;
type NvrtcGetPtXSize = unsafe extern "C" fn(u64, *mut usize) -> i32;
type NvrtcGetPtX = unsafe extern "C" fn(u64, *mut c_char) -> i32;
type NvrtcGetProgramLogSize = unsafe extern "C" fn(u64, *mut usize) -> i32;
type NvrtcGetProgramLog = unsafe extern "C" fn(u64, *mut c_char) -> i32;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut u64) -> i32;

const MATVEC_KERNEL: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void matvec_f32(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float4* row = (const float4*)(W + (size_t)r * cols);
    const float4* x4 = (const float4*)x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int c = 0;
    for (; c + 2 <= cols / 4; c += 2) {
        float4 a = row[c], b = x4[c];
        acc.x += a.x * b.x; acc.y += a.y * b.y;
        acc.z += a.z * b.z; acc.w += a.w * b.w;
        float4 a2 = row[c + 1], b2 = x4[c + 1];
        acc.x += a2.x * b2.x; acc.y += a2.y * b2.y;
        acc.z += a2.z * b2.z; acc.w += a2.w * b2.w;
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    for (; c < cols / 4; ++c) {
        float4 a = row[c], b = x4[c];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    const float* rowf = W + (size_t)r * cols;
    for (int t = (cols / 4) * 4; t < cols; ++t) sum += rowf[t] * x[t];
    y[r] = sum;
}
// Coalesced variant for tall matrices (e.g. the 152k-row output head): one
// block per row, threads strided over columns with a shared-memory reduction.
extern "C" __global__ void matvec_tall(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    const float* row = W + (size_t)r * cols;
    for (int c = tid; c < cols; c += blockDim.x) sum += row[c] * x[c];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = partial[0];
}

// ---- Quantized matvec kernels: dequantize blocks on the fly ----

__device__ __forceinline__ float cortex_h2f(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 31u;
    unsigned int mant = h & 1023u;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            int e = -14;
            unsigned int m = mant;
            while ((m & 1024u) == 0) { m <<= 1; --e; }
            bits = (sign << 31) | ((unsigned int)(e + 127) << 23) | ((m & 1023u) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return __int_as_float((int) bits);
}

__device__ __forceinline__ unsigned short ld16(const unsigned char* p) {
    return (unsigned short) p[0] | ((unsigned short) p[1] << 8);
}

__device__ __forceinline__ void get_sm_k4(int j, const unsigned char* q, int* s, int* m) {
    if (j < 4) { *s = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *s = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void matvec_q4_k(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned char* row = W + (size_t) r * (((cols + 255) / 256) * 144);
    float sum = 0.f;
    for (int b = tid; b < ((cols + 255) / 256); b += blockDim.x) {
        int c = b * 256;
        const unsigned char* blk = row + b * 144;
        float d = cortex_h2f(ld16(blk));
        float dm = cortex_h2f(ld16(blk + 2));
        const unsigned char* sc = blk + 4;
        const unsigned char* q = blk + 16;
        int is = 0;
        for (int j = 0; j < 4; ++j) {
            int s1, m1, s2, m2;
            get_sm_k4(is, sc, &s1, &m1);
            get_sm_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, m1f = dm * m1;
            float d2 = d * s2, m2f = dm * m2;
            for (int l = 0; l < 32; ++l) {
                sum += (d1 * (q[l] & 15) - m1f) * x[c + l];
                sum += (d2 * (q[l] >> 4) - m2f) * x[c + 32 + l];
            }
            q += 32; is += 2; c += 64;
        }
    }
    __shared__ float acc[128];
    acc[tid] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = acc[0];
}

extern "C" __global__ void matvec_q6_k(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned char* row = W + (size_t) r * (((cols + 255) / 256) * 210);
    float sum = 0.f;
    for (int b = tid; b < ((cols + 255) / 256); b += blockDim.x) {
        int c = b * 256;
        const unsigned char* blk = row + b * 210;
        float d = cortex_h2f(ld16(blk + 208));
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = (const signed char*) (blk + 192);
        for (int n = 0; n < 2; ++n) {
            const unsigned char* l = ql + n * 64;
            const unsigned char* h = qh + n * 32;
            const signed char* s = sc + n * 8;
            for (int i = 0; i < 32; ++i) {
                int is = i / 16;
                float q1 = (float)(int)((signed char)((l[i] & 15) | ((h[i] & 3) << 4)) - 32);
                float q2 = (float)(int)((signed char)((l[i + 32] & 15) | (((h[i] >> 2) & 3) << 4)) - 32);
                float q3 = (float)(int)((signed char)((l[i] >> 4) | (((h[i] >> 4) & 3) << 4)) - 32);
                float q4 = (float)(int)((signed char)((l[i + 32] >> 4) | (((h[i] >> 6) & 3) << 4)) - 32);
                sum += d * s[is]     * q1 * x[c + i];
                sum += d * s[is + 2] * q2 * x[c + i + 32];
                sum += d * s[is + 4] * q3 * x[c + i + 64];
                sum += d * s[is + 6] * q4 * x[c + i + 96];
            }
            c += 128;
        }
    }
    __shared__ float acc[128];
    acc[tid] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = acc[0];
}

extern "C" __global__ void matvec_q5_k(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned char* row = W + (size_t) r * (((cols + 255) / 256) * 176);
    float sum = 0.f;
    for (int b = tid; b < ((cols + 255) / 256); b += blockDim.x) {
        int c = b * 256;
        const unsigned char* blk = row + b * 176;
        float d = cortex_h2f(ld16(blk));
        float dm = cortex_h2f(ld16(blk + 2));
        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 48;
        int is = 0;
        unsigned int u1 = 1u, u2 = 2u;
        for (int j = 0; j < 4; ++j) {
            int s1, m1, s2, m2;
            get_sm_k4(is, sc, &s1, &m1);
            get_sm_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, m1f = dm * m1;
            float d2 = d * s2, m2f = dm * m2;
            for (int l = 0; l < 32; ++l) {
                float q1 = (float)((ql[l] & 15) | ((qh[l] & u1) ? 16 : 0));
                float q2 = (float)((ql[l] >> 4) | ((qh[l] & u2) ? 16 : 0));
                sum += (d1 * q1 - m1f) * x[c + l];
                sum += (d2 * q2 - m2f) * x[c + 32 + l];
            }
            ql += 32; is += 2; c += 64;
            u1 <<= 2; u2 <<= 2;
        }
    }
    __shared__ float acc[128];
    acc[tid] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = acc[0];
}

extern "C" __global__ void matvec_q5_0(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned char* row = W + (size_t) r * (((cols + 31) / 32) * 22);
    float sum = 0.f;
    for (int b = tid; b < ((cols + 31) / 32); b += blockDim.x) {
        const unsigned char* blk = row + b * 22;
        float d = cortex_h2f(ld16(blk));
        unsigned int qh = (unsigned int) blk[2] | ((unsigned int) blk[3] << 8)
                        | ((unsigned int) blk[4] << 16) | ((unsigned int) blk[5] << 24);
        for (int j = 0; j < 16; ++j) {
            unsigned int xh0 = ((qh >> j) << 4) & 16u;
            unsigned int xh1 = (qh >> (j + 12)) & 16u;
            float q0 = (float)((int)((blk[6 + j] & 15) | xh0) - 16);
            float q1 = (float)((int)((blk[6 + j] >> 4) | xh1) - 16);
            sum += q0 * d * x[b * 32 + j];
            sum += q1 * d * x[b * 32 + 16 + j];
        }
    }
    __shared__ float acc[128];
    acc[tid] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = acc[0];
}

extern "C" __global__ void matvec_q8_0(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned char* row = W + (size_t) r * (((cols + 31) / 32) * 34);
    float sum = 0.f;
    for (int b = tid; b < ((cols + 31) / 32); b += blockDim.x) {
        const unsigned char* blk = row + b * 34;
        float d = cortex_h2f(ld16(blk));
        for (int l = 0; l < 32; ++l) {
            sum += ((signed char) blk[2 + l]) * d * x[b * 32 + l];
        }
    }
    __shared__ float acc[128];
    acc[tid] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[r] = acc[0];
}

extern "C" __global__ void matvec_f16(
    const unsigned short* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned short* row = W + (size_t) r * cols;
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) sum += cortex_h2f(row[c]) * x[c];
    y[r] = sum;
}

extern "C" __global__ void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

extern "C" // 8 rows per block, one warp per row: coalesced x loads and only rows/8
// blocks for the very tall output head.
extern "C" __global__ void matvec_f32_rows(
    const float* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * 8 + threadIdx.x / 32;
    if (r >= rows) return;
    int lane = threadIdx.x & 31;
    const float* row = W + (size_t) r * cols;
    float sum = 0.f;
    for (int c = lane * 4; c + 4 <= cols; c += 128) {
        float4 wv = *(const float4*) (row + c);
        float4 xv = *(const float4*) (x + c);
        sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[r] = sum;
}

extern "C" extern "C" __global__ void matvec_f16_rows(
    const unsigned short* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * 8 + threadIdx.x / 32;
    if (r >= rows) return;
    int lane = threadIdx.x & 31;
    const unsigned short* row = W + (size_t) r * cols;
    float sum = 0.f;
    for (int c = lane * 4; c + 4 <= cols; c += 128) {
        float2 w0 = __half22float2(*(const __half2*) (row + c));
        float2 w1 = __half22float2(*(const __half2*) (row + c + 2));
        float2 x0 = *(const float2*) (x + c);
        float2 x1 = *(const float2*) (x + c + 2);
        sum += w0.x * x0.x + w0.y * x0.y + w1.x * x1.x + w1.y * x1.y;
    }
    for (int c = (cols / 4) * 4 + lane; c < cols; c += 32) {
        sum += __half2float(__ushort_as_half(row[c])) * x[c];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[r] = sum;
}

extern "C" __global__ void cortex_rmsnorm(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int n, float eps)
{
    __shared__ float acc[256];
    int tid = threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < n; i += blockDim.x) s += x[i] * x[i];
    acc[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    float scale = rsqrtf(acc[0] / (float) n + eps);
    for (int i = tid; i < n; i += blockDim.x) y[i] = x[i] * scale * w[i];
}

extern "C" __global__ void cortex_rope_qk(
    float* __restrict__ q, float* __restrict__ k,
    const float* __restrict__ cos, const float* __restrict__ sin,
    int q_heads, int kv_heads, int hd)
{
    int half = hd / 2;
    int total = (q_heads + kv_heads) * half;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int head = tid / half;
    int i = tid % half;
    float* base = head < q_heads ? q + head * hd : k + (head - q_heads) * hd;
    float a = base[i], b = base[i + half];
    float c = cos[i], s = sin[i];
    base[i] = a * c - b * s;
    base[i + half] = b * c + a * s;
}

extern "C" __global__ void cortex_append_kv(
    float* __restrict__ kc, float* __restrict__ vc,
    const float* __restrict__ k, const float* __restrict__ v,
    const int* __restrict__ pos_p, int width)
{
    int pos = *pos_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    kc[(size_t) pos * width + i] = k[i];
    vc[(size_t) pos * width + i] = v[i];
}

extern "C" __global__ void cortex_add_bias(float* __restrict__ x, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i];
}

extern "C" __global__ void cortex_add(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void cortex_attention(
    const float* __restrict__ q,
    const float* __restrict__ kc, const float* __restrict__ vc,
    float* __restrict__ out,
    const int* __restrict__ pos_p, int n_heads, int n_kv, int hd, int width, int window)
{
    int pos = *pos_p;
    extern __shared__ float share[];
    float* scores = share;
    int h = blockIdx.x;
    int kv_head = h / (n_heads / n_kv);
    const float* qh = q + h * hd;
    int start = window > 0 ? max(0, pos - window + 1) : 0;
    int count = pos - start + 1;
    int tid = threadIdx.x;
    for (int t = start + tid; t <= pos; t += blockDim.x) {
        const float* kh = kc + (size_t) t * width + kv_head * hd;
        float s = 0.f;
        for (int d = 0; d < hd; ++d) s += qh[d] * kh[d];
        scores[t - start] = s * rsqrtf((float) hd);
    }
    __syncthreads();
    float local_max = -1e30f;
    for (int i = tid; i < count; i += blockDim.x) local_max = fmaxf(local_max, scores[i]);
    share[512 + tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) share[512 + tid] = fmaxf(share[512 + tid], share[512 + tid + stride]);
        __syncthreads();
    }
    float mx = share[512];
    if (tid == 0) share[640] = 0.f;
    __syncthreads();
    float local_sum = 0.f;
    for (int i = tid; i < count; i += blockDim.x) {
        float e = __expf(scores[i] - mx);
        scores[i] = e;
        local_sum += e;
    }
    atomicAdd(&share[640], local_sum);
    __syncthreads();
    float inv = 1.f / fmaxf(share[640], 1e-20f);
    float* oh = out + h * hd;
    for (int d = tid; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int t = start; t <= pos; ++t) {
            const float* vh = vc + (size_t) t * width + kv_head * hd;
            acc += scores[t - start] * inv * vh[d];
        }
        oh[d] = acc;
    }
}
"#;

/// A buffer allocated in device memory.
pub struct CudaBuffer {
    ptr: u64,
}

impl CudaBuffer {
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // The device outlives buffers in practice; freeing after context
        // destruction is harmless.
        unsafe { LETS_FREE(self.ptr) };
    }
}

// Set by CudaDevice::new so buffers can free themselves. The function pointer
// is process-wide; multiple devices are not supported (see README_CUDA).
static mut LETS_FREE: fn(u64) = |_| {};

/// Quantized weight matrix stored in device memory.
pub struct QuantBuffer {
    ptr: u64,
    pub kind: u32,
    pub rows: usize,
    pub cols: usize,
}

struct Api {
    function: u64,
    tall: u64,
    q4k: u64,
    q6k: u64,
    q5k: u64,
    q8_0: u64,
    q5_0: u64,
    f16: u64,
    f32_rows: u64,
    f16_rows: u64,
    swiglu: u64,
    norm: u64,
    rope: u64,
    append: u64,
    add: u64,
    bias: u64,
    attn: u64,
    stream: u64,
    graph: Mutex<Option<u64>>,
    begin: CuStreamBeginCapture,
    end: CuStreamEndCapture,
    graph_inst: CuGraphInstantiate,
    graph_launch: CuGraphLaunch,
    stream_sync: CuStreamSynchronize,
    memcpy_async: CuMemcpyHtoDAsync,
    cu_mem_alloc: CuMemAlloc,
    cu_memcpy_h2d: CuMemcpyHtoD,
    cu_memcpy_d2h: CuMemcpyDtoH,
    cu_launch: CuLaunchKernel,
    cu_sync: CuCtxSynchronize,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// Initialized CUDA device with the JIT-compiled `matvec` kernel.
pub struct CudaDevice {
    api: Arc<Api>,
    /// Reused scratch buffers keyed by `(x_len, out_len)`; slot 0 is `x`.
    scratch: Mutex<HashMap<(usize, usize), Vec<CudaBuffer>>>,
    matvec_ns: AtomicU64,
    matvec_calls: AtomicU64,
    by_shape: Mutex<HashMap<(usize, usize), (u64, u64)>>,
}

impl CudaDevice {
    pub fn new() -> Result<Self> {
        let cuda = unsafe { dlopen(b"libcuda.so.1\0".as_ptr() as *const c_char, RTLD_NOW) };
        if cuda.is_null() {
            return Err(Error::InvalidConfig("libcuda.so.1 not found".into()));
        }
        let nvrtc_names = [
            "libnvrtc.so.12",
            "libnvrtc.so",
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/usr/lib/x86_64-linux-gnu/libnvrtc.so.12",
        ];
        let mut nvrtc = std::ptr::null_mut();
        for name in nvrtc_names {
            let cname = CString::new(name).unwrap();
            let h = unsafe { dlopen(cname.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                nvrtc = h;
                break;
            }
        }
        if nvrtc.is_null() {
            return Err(Error::InvalidConfig(
                "libnvrtc not found (install CUDA Toolkit)".into(),
            ));
        }

        unsafe {
            let cu_init: CuInit = ffi!(cuda, "cuInit", CuInit);
            let cu_device_get: CuDeviceGet = ffi!(cuda, "cuDeviceGet", CuDeviceGet);
            let cu_ctx_create: CuCtxCreate = ffi!(cuda, "cuCtxCreate_v2", CuCtxCreate);
            let cu_module_load: CuModuleLoadData = ffi!(cuda, "cuModuleLoadData", CuModuleLoadData);
            let cu_module_fn: CuModuleGetFunction =
                ffi!(cuda, "cuModuleGetFunction", CuModuleGetFunction);
            let cu_mem_alloc: CuMemAlloc = ffi!(cuda, "cuMemAlloc_v2", CuMemAlloc);
            let cu_mem_free: CuMemFree = ffi!(cuda, "cuMemFree_v2", CuMemFree);
            let cu_memcpy_h2d: CuMemcpyHtoD = ffi!(cuda, "cuMemcpyHtoD_v2", CuMemcpyHtoD);
            let cu_memcpy_d2h: CuMemcpyDtoH = ffi!(cuda, "cuMemcpyDtoH_v2", CuMemcpyDtoH);
            let cu_launch: CuLaunchKernel = ffi!(cuda, "cuLaunchKernel", CuLaunchKernel);
            let cu_sync: CuCtxSynchronize = ffi!(cuda, "cuCtxSynchronize", CuCtxSynchronize);
            let cu_stream_create: CuStreamCreate = ffi!(cuda, "cuStreamCreate", CuStreamCreate);
            let cu_begin: CuStreamBeginCapture = ffi!(cuda, "cuStreamBeginCapture", CuStreamBeginCapture);
            let cu_end: CuStreamEndCapture = ffi!(cuda, "cuStreamEndCapture", CuStreamEndCapture);
            let cu_graph_inst: CuGraphInstantiate = ffi!(cuda, "cuGraphInstantiate_v2", CuGraphInstantiate);
            let cu_graph_launch: CuGraphLaunch = ffi!(cuda, "cuGraphLaunch", CuGraphLaunch);
            let cu_stream_sync: CuStreamSynchronize = ffi!(cuda, "cuStreamSynchronize", CuStreamSynchronize);
            let cu_memcpy_async: CuMemcpyHtoDAsync = ffi!(cuda, "cuMemcpyHtoDAsync_v2", CuMemcpyHtoDAsync);

            let nvrtc_create: NvrtcCreateProgram = ffi!(nvrtc, "nvrtcCreateProgram", NvrtcCreateProgram);
            let nvrtc_compile: NvrtcCompileProgram = ffi!(nvrtc, "nvrtcCompileProgram", NvrtcCompileProgram);
            let nvrtc_ptx_size: NvrtcGetPtXSize = ffi!(nvrtc, "nvrtcGetPTXSize", NvrtcGetPtXSize);
            let nvrtc_ptx: NvrtcGetPtX = ffi!(nvrtc, "nvrtcGetPTX", NvrtcGetPtX);
            let nvrtc_log_size: NvrtcGetProgramLogSize =
                ffi!(nvrtc, "nvrtcGetProgramLogSize", NvrtcGetProgramLogSize);
            let nvrtc_log: NvrtcGetProgramLog = ffi!(nvrtc, "nvrtcGetProgramLog", NvrtcGetProgramLog);
            let nvrtc_destroy: NvrtcDestroyProgram = ffi!(nvrtc, "nvrtcDestroyProgram", NvrtcDestroyProgram);

            check(cu_init(0), "cuInit")?;
            let mut device = 0;
            check(cu_device_get(&mut device, 0), "cuDeviceGet")?;
            let mut ctx = 0u64;
            check(cu_ctx_create(&mut ctx, 0, device), "cuCtxCreate")?;

            let src = CString::new(MATVEC_KERNEL).unwrap();
            let name = CString::new("matvec.cu").unwrap();
            let mut prog = 0u64;
            check(
                nvrtc_create(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()),
                "nvrtcCreateProgram",
            )?;
            let inc = CString::new("--include-path=/usr/include").unwrap();
            let opts = [inc.as_ptr()];
            let compile = nvrtc_compile(prog, 1, opts.as_ptr());
            if compile != CUDA_SUCCESS {
                let mut size = 0usize;
                nvrtc_log_size(prog, &mut size);
                let mut log = vec![0i8; size + 1];
                nvrtc_log(prog, log.as_mut_ptr());
                nvrtc_destroy(&mut prog);
                let msg = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
                return Err(Error::InvalidConfig(format!("NVRTC compile failed: {msg}")));
            }
            let mut ptx_size = 0usize;
            check(nvrtc_ptx_size(prog, &mut ptx_size), "nvrtcGetPTXSize")?;
            let mut ptx = vec![0i8; ptx_size + 1];
            check(nvrtc_ptx(prog, ptx.as_mut_ptr()), "nvrtcGetPTX")?;
            nvrtc_destroy(&mut prog);

            let mut module = 0u64;
            check(cu_module_load(&mut module, ptx.as_ptr() as *const c_void), "cuModuleLoadData")?;
            let fname = CString::new("matvec_f32").unwrap();
            let mut function = 0u64;
            check(cu_module_fn(&mut function, module, fname.as_ptr()), "cuModuleGetFunction")?;
            let tname = CString::new("matvec_tall").unwrap();
            let mut tall = 0u64;
            check(cu_module_fn(&mut tall, module, tname.as_ptr()), "cuModuleGetFunction(tall)")?;
            let mut kq4 = 0u64;
            let n = CString::new("matvec_q4_k").unwrap();
            check(cu_module_fn(&mut kq4, module, n.as_ptr()), "fn q4k")?;
            let mut kq6 = 0u64;
            let n = CString::new("matvec_q6_k").unwrap();
            check(cu_module_fn(&mut kq6, module, n.as_ptr()), "fn q6k")?;
            let mut kq5 = 0u64;
            let n = CString::new("matvec_q5_k").unwrap();
            check(cu_module_fn(&mut kq5, module, n.as_ptr()), "fn q5k")?;
            let mut kq8 = 0u64;
            let n = CString::new("matvec_q8_0").unwrap();
            check(cu_module_fn(&mut kq8, module, n.as_ptr()), "fn q8_0")?;
            let mut kq50 = 0u64;
            let n = CString::new("matvec_q5_0").unwrap();
            check(cu_module_fn(&mut kq50, module, n.as_ptr()), "fn q5_0")?;
            let mut kf16 = 0u64;
            let n = CString::new("matvec_f16").unwrap();
            check(cu_module_fn(&mut kf16, module, n.as_ptr()), "fn f16")?;
            let mut hrows = 0u64;
            let n = CString::new("matvec_f32_rows").unwrap();
            check(cu_module_fn(&mut hrows, module, n.as_ptr()), "fn rows")?;
            let mut hf16r = 0u64;
            let n = CString::new("matvec_f16_rows").unwrap();
            check(cu_module_fn(&mut hf16r, module, n.as_ptr()), "fn f16_rows")?;
            let mut hnorm = 0u64;
            let n = CString::new("cortex_rmsnorm").unwrap();
            check(cu_module_fn(&mut hnorm, module, n.as_ptr()), "fn norm")?;
            let mut hrope = 0u64;
            let n = CString::new("cortex_rope_qk").unwrap();
            check(cu_module_fn(&mut hrope, module, n.as_ptr()), "fn rope")?;
            let mut happ = 0u64;
            let n = CString::new("cortex_append_kv").unwrap();
            check(cu_module_fn(&mut happ, module, n.as_ptr()), "fn append")?;
            let mut hbias = 0u64;
            let n = CString::new("cortex_add_bias").unwrap();
            check(cu_module_fn(&mut hbias, module, n.as_ptr()), "fn bias")?;
            let mut hadd = 0u64;
            let n = CString::new("cortex_add").unwrap();
            check(cu_module_fn(&mut hadd, module, n.as_ptr()), "fn add")?;
            let mut hatt = 0u64;
            let n = CString::new("cortex_attention").unwrap();
            check(cu_module_fn(&mut hatt, module, n.as_ptr()), "fn attn")?;
            let swname = CString::new("swiglu_kernel").unwrap();
            let mut swiglu = 0u64;
            check(cu_module_fn(&mut swiglu, module, swname.as_ptr()), "cuModuleGetFunction(swiglu)")?;

            let mut stream = 0u64;
            check(cu_stream_create(&mut stream, 0), "cuStreamCreate")?;
            if std::env::var_os("CORTEX_CUDA_DEBUG").is_some() { eprintln!("stream={stream}, ctx={ctx}, device={device}"); }

            LETS_FREE = std::mem::transmute::<CuMemFree, fn(u64)>(cu_mem_free);

            Ok(Self {
                api: Arc::new(Api {
                    function,
                    tall,
                    q4k: kq4,
                    q6k: kq6,
                    q5k: kq5,
                    q8_0: kq8,
                    q5_0: kq50,
                    f16: kf16,
                    f32_rows: hrows,
                    f16_rows: hf16r,
                    swiglu,
                    norm: hnorm,
                    rope: hrope,
                    append: happ,
                    add: hadd,
                    bias: hbias,
                    attn: hatt,
                    stream,
                    graph: Mutex::new(None),
                    begin: cu_begin,
                    end: cu_end,
                    graph_inst: cu_graph_inst,
                    graph_launch: cu_graph_launch,
                    stream_sync: cu_stream_sync,
                    memcpy_async: cu_memcpy_async,
                    cu_mem_alloc,
                    cu_memcpy_h2d,
                    cu_memcpy_d2h,
                    cu_launch,
                    cu_sync,
                }),
                scratch: Mutex::new(HashMap::new()),
                matvec_ns: AtomicU64::new(0),
                matvec_calls: AtomicU64::new(0),
                by_shape: Mutex::new(HashMap::new()),
            })
        }
    }

    pub fn upload(&self, data: &[f32]) -> Result<CudaBuffer> {
        let bytes = data.len() * std::mem::size_of::<f32>();
        let mut ptr = 0u64;
        unsafe {
            check((self.api.cu_mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
            check(
                (self.api.cu_memcpy_h2d)(ptr, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(CudaBuffer { ptr })
    }

    /// Allocate an uninitialized device buffer of `len` floats.
    pub fn alloc(&self, len: usize) -> Result<CudaBuffer> {
        let mut ptr = 0u64;
        unsafe {
            check(
                (self.api.cu_mem_alloc)(&mut ptr, len * 4),
                "cuMemAlloc",
            )?;
        }
        Ok(CudaBuffer { ptr })
    }

    pub fn copy_to_device(&self, dst: &CudaBuffer, src: &[f32]) -> Result<()> {
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(dst.ptr, src.as_ptr() as *const c_void, src.len() * 4),
                "cuMemcpyHtoD",
            )
        }
    }

    pub fn copy_to_host(&self, dst: &mut [f32], src: &CudaBuffer) -> Result<()> {
        unsafe {
            check(
                (self.api.cu_memcpy_d2h)(dst.as_mut_ptr() as *mut c_void, src.ptr, dst.len() * 4),
                "cuMemcpyDtoH",
            )
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check((self.api.cu_sync)(), "cuCtxSynchronize") }
    }

    pub fn copy_to_device_async(&self, dst: &CudaBuffer, src: &[f32]) -> Result<()> {
        unsafe {
            check(
                (self.api.memcpy_async)(dst.ptr, src.as_ptr() as *const c_void, src.len() * 4, self.api.stream),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    pub fn copy_bytes_to_device_async(&self, dst: &CudaBuffer, src: &[u8]) -> Result<()> {
        unsafe {
            check(
                (self.api.memcpy_async)(dst.ptr, src.as_ptr() as *const c_void, src.len(), self.api.stream),
                "cuMemcpyHtoDAsync(bytes)",
            )
        }
    }

    pub fn graph_begin(&self) -> Result<()> {
        unsafe { check((self.api.begin)(self.api.stream, 0), "cuStreamBeginCapture") }
    }

    pub fn graph_end(&self) -> Result<()> {
        let mut graph = 0u64;
        unsafe {
            check((self.api.end)(self.api.stream, &mut graph), "cuStreamEndCapture")?;
            let mut exec = 0u64;
            check(
                (self.api.graph_inst)(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0, 0),
                "cuGraphInstantiate",
            )?;
            *self.api.graph.lock().unwrap() = Some(exec);
        }
        Ok(())
    }

    pub fn graph_ready(&self) -> bool {
        self.api.graph.lock().unwrap().is_some()
    }

    pub fn graph_launch(&self) -> Result<()> {
        let exec = self.api.graph.lock().unwrap().unwrap();
        unsafe { check((self.api.graph_launch)(exec, self.api.stream), "cuGraphLaunch") }
    }

    pub fn stream_sync(&self) -> Result<()> {
        unsafe { check((self.api.stream_sync)(self.api.stream), "cuStreamSynchronize") }
    }

    pub fn rmsnorm_device(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        y: &CudaBuffer,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let mut p: [*mut c_void; 5] = [
            &x.ptr as *const u64 as *mut c_void,
            &w.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &(n as i32) as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.norm, 1, 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch rmsnorm",
            )
        }
    }

    pub fn rope_device(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        q_heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Result<()> {
        let total = (q_heads + kv_heads) * (hd / 2);
        let mut p: [*mut c_void; 7] = [
            &q.ptr as *const u64 as *mut c_void,
            &k.ptr as *const u64 as *mut c_void,
            &cos.ptr as *const u64 as *mut c_void,
            &sin.ptr as *const u64 as *mut c_void,
            &(q_heads as i32) as *const i32 as *mut c_void,
            &(kv_heads as i32) as *const i32 as *mut c_void,
            &(hd as i32) as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.rope, (total as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch rope",
            )
        }
    }

    pub fn append_kv_device(
        &self,
        kc: &CudaBuffer,
        vc: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        pos: &CudaBuffer,
        width: usize,
    ) -> Result<()> {
        let mut p: [*mut c_void; 6] = [
            &kc.ptr as *const u64 as *mut c_void,
            &vc.ptr as *const u64 as *mut c_void,
            &k.ptr as *const u64 as *mut c_void,
            &v.ptr as *const u64 as *mut c_void,
            &pos.ptr as *const u64 as *mut c_void,
            &(width as i32) as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.append, (width as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch append_kv",
            )
        }
    }

    pub fn add_bias_device(&self, x: &CudaBuffer, b: &CudaBuffer, n: usize) -> Result<()> {
        let mut p: [*mut c_void; 3] = [
            &x.ptr as *const u64 as *mut c_void,
            &b.ptr as *const u64 as *mut c_void,
            &(n as i32) as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.bias, (n as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch add_bias",
            )
        }
    }

    pub fn add_device(&self, a: &CudaBuffer, b: &CudaBuffer, y: &CudaBuffer, n: usize) -> Result<()> {
        let mut p: [*mut c_void; 4] = [
            &a.ptr as *const u64 as *mut c_void,
            &b.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &(n as i32) as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.add, (n as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch add",
            )
        }
    }

    pub fn attention_device(
        &self,
        q: &CudaBuffer,
        kc: &CudaBuffer,
        vc: &CudaBuffer,
        out: &CudaBuffer,
        pos: &CudaBuffer,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        width: usize,
        window: usize,
    ) -> Result<()> {
        let mut p: [*mut c_void; 10] = [
            &q.ptr as *const u64 as *mut c_void,
            &kc.ptr as *const u64 as *mut c_void,
            &vc.ptr as *const u64 as *mut c_void,
            &out.ptr as *const u64 as *mut c_void,
            &pos.ptr as *const u64 as *mut c_void,
            &(n_heads as i32) as *const i32 as *mut c_void,
            &(n_kv as i32) as *const i32 as *mut c_void,
            &(hd as i32) as *const i32 as *mut c_void,
            &(width as i32) as *const i32 as *mut c_void,
            &(window as i32) as *const i32 as *mut c_void,
        ];
        // 512 scores + max + sum slots
        let shared = (512 + 128 + 1) * 4;
        unsafe {
            check(
                (self.api.cu_launch)(self.api.attn, n_heads as u32, 1, 1, 128, 1, 1, shared, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch attention",
            )
        }
    }

    pub fn swiglu_device(&self, gate: &CudaBuffer, up: &CudaBuffer, out: &CudaBuffer, n: usize) -> Result<()> {
        let mut p: [*mut c_void; 4] = [
            &gate.ptr as *const u64 as *mut c_void,
            &up.ptr as *const u64 as *mut c_void,
            &out.ptr as *const u64 as *mut c_void,
            &(n as i32) as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.swiglu, (n as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch swiglu",
            )
        }
    }

    /// Launch a quantized matvec kernel without synchronizing.
    pub fn matvec_q_device(
        &self,
        w: &QuantBuffer,
        x: &CudaBuffer,
        y: &CudaBuffer,
    ) -> Result<()> {
        let func = match w.kind {
            12 => self.api.q4k,
            14 => self.api.q6k,
            13 => self.api.q5k,
            8 => self.api.q8_0,
            6 => self.api.q5_0,
            1 => self.api.f16,
            0 => self.api.function,
            other => return Err(Error::InvalidConfig(format!("CUDA: unsupported type {other}"))),
        };
        let rows = w.rows as i32;
        let cols = w.cols as i32;
        let mut p: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void,
            &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &rows as *const i32 as *mut c_void,
            &cols as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(func, w.rows as u32, 1, 1, 128, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch quantized matvec",
            )
        }
    }

    /// Launch the f32 matvec kernel without synchronizing (uses `matvec_f32`).
    /// Coalesced block-per-row f32 matvec (for the very tall output head).
    pub fn matvec_f16_rows_device(&self, w: &QuantBuffer, x: &CudaBuffer, y: &CudaBuffer) -> Result<()> {
        let r = w.rows as i32;
        let c = w.cols as i32;
        let mut p: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void,
            &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &r as *const i32 as *mut c_void,
            &c as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.f16_rows, (w.rows as u32).div_ceil(8), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch f16 rows",
            )
        }
    }

    pub fn matvec_f32_tall_device(&self, w: &CudaBuffer, x: &CudaBuffer, y: &CudaBuffer, rows: usize, cols: usize) -> Result<()> {
        let r = rows as i32;
        let c = cols as i32;
        let mut p: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void,
            &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &r as *const i32 as *mut c_void,
            &c as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.f32_rows, (rows as u32).div_ceil(8), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch tall matvec",
            )
        }
    }

    pub fn matvec_f32_device(&self, w: &CudaBuffer, x: &CudaBuffer, y: &CudaBuffer, rows: usize, cols: usize) -> Result<()> {
        let r = rows as i32;
        let c = cols as i32;
        let mut p: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void,
            &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void,
            &r as *const i32 as *mut c_void,
            &c as *const i32 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.cu_launch)(self.api.function, (rows as u32).div_ceil(256), 1, 1, 256, 1, 1, 0, self.api.stream, p.as_mut_ptr(), std::ptr::null_mut()),
                "launch f32 matvec",
            )
        }
    }

    /// `(total nanoseconds, number of calls)` spent in `matvec` so far.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.matvec_ns.load(Ordering::Relaxed),
            self.matvec_calls.load(Ordering::Relaxed),
        )
    }

    /// Per-shape `(rows, cols) -> (total ns, calls)` statistics.
    pub fn stats_by_shape(&self) -> HashMap<(usize, usize), (u64, u64)> {
        self.by_shape.lock().unwrap().clone()
    }

    /// Run several `out = W_i * x` products with a single synchronization.
    ///
    /// `slot` indexes into per-shape scratch buffers, so callers can group
    /// q/k/v or gate/up without extra allocations.
    pub fn matvec_many(
        &self,
        weights: &[&CudaBuffer],
        x: &[f32],
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        let started = Instant::now();
        if weights.is_empty() || x.is_empty() {
            return Ok(());
        }
        let cols = x.len() as i32;
        let mut cache = self.scratch.lock().unwrap();
        let xkey = (x.len(), 0usize);
        let bufs = cache.entry(xkey).or_default();
        if bufs.is_empty() {
            bufs.push(self.upload(&vec![0.0f32; x.len()])?);
        }
        let x_ptr = bufs[0].ptr;
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(x_ptr, x.as_ptr() as *const c_void, x.len() * 4),
                "cuMemcpyHtoD(x)",
            )?;
        }

        let threads = 256u32;
        let mut pending: Vec<(u64, *mut f32, usize)> = Vec::with_capacity(weights.len());
        let mut slot = 0usize;
        for (w, out) in weights.iter().zip(outs.iter_mut()) {
            if out.is_empty() {
                continue;
            }
            let rows = out.len() as i32;
            let ykey = (x.len(), out.len());
            let bufs = cache.entry(ykey).or_default();
            while bufs.len() <= slot {
                bufs.push(self.upload(&vec![0.0f32; out.len()])?);
            }
            let y_ptr = bufs[slot].ptr;
            unsafe {
                let mut params: [*mut c_void; 5] = [
                    &w.ptr as *const u64 as *mut c_void,
                    &x_ptr as *const u64 as *mut c_void,
                    &y_ptr as *const u64 as *mut c_void,
                    &rows as *const i32 as *mut c_void,
                    &cols as *const i32 as *mut c_void,
                ];
                let blocks = (out.len() as u32).div_ceil(threads);
                check(
                    (self.api.cu_launch)(
                        self.api.function,
                        blocks,
                        1,
                        1,
                        threads,
                        1,
                        1,
                        0,
                        self.api.stream,
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "cuLaunchKernel",
                )?;
            }
            pending.push((y_ptr, out.as_mut_ptr(), out.len()));
            slot += 1;
        }

        unsafe { check((self.api.cu_sync)(), "cuCtxSynchronize")? };
        for (y, host, len) in pending {
            unsafe {
                check(
                    (self.api.cu_memcpy_d2h)(host as *mut c_void, y, len * 4),
                    "cuMemcpyDtoH",
                )?;
            }
        }

        let elapsed = started.elapsed().as_nanos() as u64;
        self.matvec_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.matvec_calls.fetch_add(weights.len() as u64, Ordering::Relaxed);
        let mut shapes = self.by_shape.lock().unwrap();
        for out in outs.iter() {
            let entry = shapes.entry((out.len(), x.len())).or_insert((0, 0));
            entry.0 += elapsed;
            entry.1 += 1;
        }
        Ok(())
    }

    pub fn matvec(&self, weights: &CudaBuffer, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.matvec_many(&[weights], x, &mut [out])
    }

    /// Coalesced matvec for very tall matrices (the output head).
    pub fn matvec_tall(&self, weights: &CudaBuffer, x: &[f32], out: &mut [f32]) -> Result<()> {
        let started = Instant::now();
        let mut cache = self.scratch.lock().unwrap();
        let xkey = (x.len(), 0usize);
        let xbufs = cache.entry(xkey).or_default();
        if xbufs.is_empty() {
            xbufs.push(self.upload(&vec![0.0f32; x.len()])?);
        }
        let x_ptr = xbufs[0].ptr;
        let ykey = (x.len(), out.len());
        let ybufs = cache.entry(ykey).or_default();
        if ybufs.is_empty() {
            ybufs.push(self.upload(&vec![0.0f32; out.len()])?);
        }
        let y_ptr = ybufs[0].ptr;
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(x_ptr, x.as_ptr() as *const c_void, x.len() * 4),
                "cuMemcpyHtoD(x)",
            )?;
            let cols = x.len() as i32;
            let mut params: [*mut c_void; 4] = [
                &weights.ptr as *const u64 as *mut c_void,
                &x_ptr as *const u64 as *mut c_void,
                &y_ptr as *const u64 as *mut c_void,
                &cols as *const i32 as *mut c_void,
            ];
            check(
                (self.api.cu_launch)(
                    self.api.tall,
                    out.len() as u32,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    self.api.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(tall)",
            )?;
            check((self.api.cu_sync)(), "cuCtxSynchronize")?;
            check(
                (self.api.cu_memcpy_d2h)(out.as_mut_ptr() as *mut c_void, y_ptr, out.len() * 4),
                "cuMemcpyDtoH",
            )?;
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        self.matvec_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.matvec_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Upload raw quantized tensor bytes to VRAM.
    pub fn upload_quant(
        &self,
        kind: u32,
        rows: usize,
        cols: usize,
        data: &[u8],
    ) -> Result<QuantBuffer> {
        let mut ptr = 0u64;
        unsafe {
            check(
                (self.api.cu_mem_alloc)(&mut ptr, data.len()),
                "cuMemAlloc(quant)",
            )?;
            check(
                (self.api.cu_memcpy_h2d)(ptr, data.as_ptr() as *const c_void, data.len()),
                "cuMemcpyHtoD(quant)",
            )?;
        }
        Ok(QuantBuffer { ptr, kind, rows, cols })
    }

    /// `out = quantized_weight * x`, dequantizing each block on the GPU.
    pub fn matvec_quant(&self, w: &QuantBuffer, x: &[f32], out: &mut [f32]) -> Result<()> {
        let func = match w.kind {
            12 => self.api.q4k,
            14 => self.api.q6k,
            13 => self.api.q5k,
            8 => self.api.q8_0,
            0 => self.api.function,
            1 => self.api.f16,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "CUDA: unsupported quantization type {other}"
                )))
            }
        };
        let started = Instant::now();
        let mut cache = self.scratch.lock().unwrap();
        let xkey = (x.len(), 0usize);
        let xbufs = cache.entry(xkey).or_default();
        if xbufs.is_empty() {
            xbufs.push(self.upload(&vec![0.0f32; x.len()])?);
        }
        let x_ptr = xbufs[0].ptr;
        let ykey = (x.len(), w.rows);
        let ybufs = cache.entry(ykey).or_default();
        if ybufs.is_empty() {
            ybufs.push(self.upload(&vec![0.0f32; w.rows])?);
        }
        let y_ptr = ybufs[0].ptr;
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(x_ptr, x.as_ptr() as *const c_void, x.len() * 4),
                "cuMemcpyHtoD(x)",
            )?;
            let rows = w.rows as i32;
            let cols = w.cols as i32;
            let mut params: [*mut c_void; 5] = [
                &w.ptr as *const u64 as *mut c_void,
                &x_ptr as *const u64 as *mut c_void,
                &y_ptr as *const u64 as *mut c_void,
                &rows as *const i32 as *mut c_void,
                &cols as *const i32 as *mut c_void,
            ];
            check(
                (self.api.cu_launch)(
                    func,
                    w.rows as u32,
                    1, 1, 128, 1, 1, 0,
                    self.api.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(quant)",
            )?;
            check((self.api.cu_sync)(), "cuCtxSynchronize")?;
            check(
                (self.api.cu_memcpy_d2h)(out.as_mut_ptr() as *mut c_void, y_ptr, out.len() * 4),
                "cuMemcpyDtoH",
            )?;
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        self.matvec_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.matvec_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Complete SwiGLU feed-forward block in one synchronization:
    /// `gate = Wg*x`, `up = Wu*x`, `act = silu(gate)*up`, `out = Wd*act`.
    pub fn ffn(
        &self,
        w_gate: &CudaBuffer,
        w_up: &CudaBuffer,
        w_down: &CudaBuffer,
        x: &[f32],
        out: &mut [f32],
        ffn_len: usize,
    ) -> Result<()> {
        let started = Instant::now();
        let cols = x.len() as i32;
        let mut cache = self.scratch.lock().unwrap();

        let xkey = (x.len(), 0usize);
        let xbufs = cache.entry(xkey).or_default();
        if xbufs.is_empty() {
            xbufs.push(self.upload(&vec![0.0f32; x.len()])?);
        }
        let x_ptr = xbufs[0].ptr;
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(x_ptr, x.as_ptr() as *const c_void, x.len() * 4),
                "cuMemcpyHtoD(x)",
            )?;
        }

        let threads = 256u32;
        let mut ensure = |key: (usize, usize), slot: usize| -> Result<u64> {
            let bufs = cache.entry(key).or_default();
            while bufs.len() <= slot {
                bufs.push(self.upload(&vec![0.0f32; key.1])?);
            }
            Ok(bufs[slot].ptr)
        };
        let gate_ptr = ensure((x.len(), ffn_len), 0)?;
        let up_ptr = ensure((x.len(), ffn_len), 1)?;
        let act_ptr = ensure((x.len(), ffn_len), 2)?;
        let down_ptr = ensure((x.len(), out.len()), 0)?;

        unsafe {
            let launch_mv = |w: &CudaBuffer, xptr: u64, yptr: u64, rows: i32, incols: i32| {
                let mut params: [*mut c_void; 5] = [
                    &w.ptr as *const u64 as *mut c_void,
                    &xptr as *const u64 as *mut c_void,
                    &yptr as *const u64 as *mut c_void,
                    &rows as *const i32 as *mut c_void,
                    &incols as *const i32 as *mut c_void,
                ];
                check(
                    (self.api.cu_launch)(
                        self.api.function,
                        (rows as u32).div_ceil(threads),
                        1, 1, threads, 1, 1, 0,
                        self.api.stream,
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(mv)",
                )
            };
            launch_mv(w_gate, x_ptr, gate_ptr, ffn_len as i32, cols)?;
            launch_mv(w_up, x_ptr, up_ptr, ffn_len as i32, cols)?;

            let mut sp: [*mut c_void; 4] = [
                &gate_ptr as *const u64 as *mut c_void,
                &up_ptr as *const u64 as *mut c_void,
                &act_ptr as *const u64 as *mut c_void,
                &(ffn_len as i32) as *const i32 as *mut c_void,
            ];
            check(
                (self.api.cu_launch)(
                    self.api.swiglu,
                    (ffn_len as u32).div_ceil(threads),
                    1, 1, threads, 1, 1, 0,
                    self.api.stream,
                    sp.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(swiglu)",
            )?;

            let incols = ffn_len as i32;
            launch_mv(w_down, act_ptr, down_ptr, out.len() as i32, incols)?;

            check((self.api.cu_sync)(), "cuCtxSynchronize")?;
            check(
                (self.api.cu_memcpy_d2h)(
                    out.as_mut_ptr() as *mut c_void,
                    down_ptr,
                    out.len() * 4,
                ),
                "cuMemcpyDtoH",
            )?;
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        self.matvec_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.matvec_calls.fetch_add(3, Ordering::Relaxed);
        let mut shapes = self.by_shape.lock().unwrap();
        let entry = shapes.entry((out.len(), x.len())).or_insert((0, 0));
        entry.0 += elapsed;
        entry.1 += 1;
        Ok(())
    }
}

fn check(code: i32, what: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::InvalidConfig(format!("{what} failed with CUDA error {code}")))
    }
}
