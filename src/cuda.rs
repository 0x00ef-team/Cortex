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
    u64, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut c_void,
) -> i32;
type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

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
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char* row = W + (size_t) r * ((cols / 256) * 144);
    float sum = 0.f;
    int c = 0;
    for (int b = 0; b < cols / 256; ++b) {
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
    y[r] = sum;
}

extern "C" __global__ void matvec_q6_k(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char* row = W + (size_t) r * ((cols / 256) * 210);
    float sum = 0.f;
    int c = 0;
    for (int b = 0; b < cols / 256; ++b) {
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
    y[r] = sum;
}

extern "C" __global__ void matvec_q5_k(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char* row = W + (size_t) r * ((cols / 256) * 176);
    float sum = 0.f;
    int c = 0;
    for (int b = 0; b < cols / 256; ++b) {
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
    y[r] = sum;
}

extern "C" __global__ void matvec_q8_0(
    const unsigned char* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned char* row = W + (size_t) r * ((cols / 32) * 34);
    float sum = 0.f;
    for (int b = 0; b < cols / 32; ++b) {
        const unsigned char* blk = row + b * 34;
        float d = cortex_h2f(ld16(blk));
        for (int l = 0; l < 32; ++l) {
            sum += ((signed char) blk[2 + l]) * d * x[b * 32 + l];
        }
    }
    y[r] = sum;
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
"#;

/// A buffer allocated in device memory.
pub struct CudaBuffer {
    ptr: u64,
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
    f16: u64,
    swiglu: u64,
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
            let compile = nvrtc_compile(prog, 0, std::ptr::null());
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
            let mut kf16 = 0u64;
            let n = CString::new("matvec_f16").unwrap();
            check(cu_module_fn(&mut kf16, module, n.as_ptr()), "fn f16")?;
            let swname = CString::new("swiglu_kernel").unwrap();
            let mut swiglu = 0u64;
            check(cu_module_fn(&mut swiglu, module, swname.as_ptr()), "cuModuleGetFunction(swiglu)")?;

            LETS_FREE = std::mem::transmute::<CuMemFree, fn(u64)>(cu_mem_free);

            Ok(Self {
                api: Arc::new(Api {
                    function,
                    tall,
                    q4k: kq4,
                    q6k: kq6,
                    q5k: kq5,
                    q8_0: kq8,
                    f16: kf16,
                    swiglu,
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
                        std::ptr::null_mut(),
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
                    std::ptr::null_mut(),
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
                    (w.rows as u32).div_ceil(256),
                    1, 1, 256, 1, 1, 0,
                    std::ptr::null_mut(),
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
                        std::ptr::null_mut(),
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
                    std::ptr::null_mut(),
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
