//! Minimal CUDA driver backend: dynamic loading, NVRTC JIT and `matvec`.
//!
//! No Rust crate dependencies — the CUDA Driver API and NVRTC are loaded at
//! runtime via `dlopen`. If the libraries are missing, `CudaDevice::new`
//! returns an error and the CPU backend can be used instead.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex};

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
"#;

/// A buffer allocated in device memory.
pub struct CudaBuffer {
    ptr: u64,
    len: usize,
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

struct Api {
    function: u64,
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
    /// Reused `x` / `y` buffers, keyed by `(x_len, out_len)`.
    scratch: Mutex<HashMap<(usize, usize), (CudaBuffer, CudaBuffer)>>,
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

            LETS_FREE = std::mem::transmute::<CuMemFree, fn(u64)>(cu_mem_free);

            Ok(Self {
                api: Arc::new(Api {
                    function,
                    cu_mem_alloc,
                    cu_memcpy_h2d,
                    cu_memcpy_d2h,
                    cu_launch,
                    cu_sync,
                }),
                scratch: Mutex::new(HashMap::new()),
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
        Ok(CudaBuffer { ptr, len: data.len() })
    }

    pub fn matvec(&self, weights: &CudaBuffer, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.is_empty() || weights.len == 0 || out.is_empty() {
            return Ok(());
        }
        let rows = out.len() as i32;
        let cols = x.len() as i32;
        let key = (x.len(), out.len());
        let mut cache = self.scratch.lock().unwrap();
        if !cache.contains_key(&key) {
            let xb = self.upload(&vec![0.0f32; x.len()])?;
            let yb = self.upload(&vec![0.0f32; out.len()])?;
            cache.insert(key, (xb, yb));
        }
        let (x_buf, y_buf) = cache.get(&key).unwrap();
        unsafe {
            check(
                (self.api.cu_memcpy_h2d)(x_buf.ptr, x.as_ptr() as *const c_void, x.len() * 4),
                "cuMemcpyHtoD(x)",
            )?;
            let mut params: [*mut c_void; 5] = [
                &weights.ptr as *const u64 as *mut c_void,
                &x_buf.ptr as *const u64 as *mut c_void,
                &y_buf.ptr as *const u64 as *mut c_void,
                &rows as *const i32 as *mut c_void,
                &cols as *const i32 as *mut c_void,
            ];
            let threads = 256u32;
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
            check((self.api.cu_sync)(), "cuCtxSynchronize")?;
            check(
                (self.api.cu_memcpy_d2h)(out.as_mut_ptr() as *mut c_void, y_buf.ptr, out.len() * 4),
                "cuMemcpyDtoH",
            )?;
        }
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
