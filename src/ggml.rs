//! Optional FFI backend to a prebuilt llama.cpp shared library.
//!
//! This is NOT part of the from-scratch CPU/CUDA engine: it is an opt-in path
//! for users who already have llama.cpp installed and want its tuned MMQ CUDA
//! kernels. No Rust crate dependencies are used - the library is loaded at
//! runtime with `dlopen`, and headers were mirrored by hand from llama.h.

use crate::error::{Error, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
use std::time::Instant;

#[repr(C, align(8))]
struct ModelParams([u8; 80]);

#[repr(C, align(8))]
struct ContextParams([u8; 176]);

#[repr(C)]
struct LlamaBatch {
    n_tokens: c_int,
    _pad: c_int,
    token: *mut c_int,
    embd: *mut f32,
    pos: *mut c_int,
    n_seq_id: *mut c_int,
    seq_id: *mut *mut c_int,
    logits: *mut i8,
}

type FnVoid = unsafe extern "C" fn();
type FnModelParams = unsafe extern "C" fn() -> ModelParams;
type FnCtxParams = unsafe extern "C" fn() -> ContextParams;
type FnLoad = unsafe extern "C" fn(*const c_char, ModelParams) -> *mut c_void;
type FnInit = unsafe extern "C" fn(*mut c_void, ContextParams) -> *mut c_void;
type FnVoidPtr = unsafe extern "C" fn(*mut c_void);
type FnPtrRet = unsafe extern "C" fn(*const c_void) -> *mut c_void;
type FnBatch = unsafe extern "C" fn(*mut c_int, c_int) -> LlamaBatch;
type FnDecode = unsafe extern "C" fn(*mut c_void, LlamaBatch) -> c_int;
type FnLogits = unsafe extern "C" fn(*mut c_void, c_int) -> *mut f32;
type FnI32 = unsafe extern "C" fn(*const c_void) -> c_int;
type FnTokenize = unsafe extern "C" fn(
    *const c_void, *const c_char, c_int, *mut c_int, c_int, bool, bool,
) -> c_int;
type FnDetok = unsafe extern "C" fn(
    *const c_void, *const c_int, c_int, *mut c_char, c_int, bool,
) -> c_int;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

unsafe fn sym<T: Copy>(h: *mut c_void, name: &str) -> Result<T> {
    let n = CString::new(name).unwrap();
    let p = dlsym(h, n.as_ptr());
    if p.is_null() {
        let e = dlerror();
        let msg = if e.is_null() {
            "unknown".into()
        } else {
            std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
        };
        return Err(Error::InvalidConfig(format!("llama.cpp symbol {name}: {msg}")));
    }
    Ok(std::mem::transmute_copy(&p))
}

/// Handle to llama.cpp library functions.
struct Lib {
    _h: *mut c_void,
    _ggml: *mut c_void,
    default_model: FnModelParams,
    default_ctx: FnCtxParams,
    load: FnLoad,
    init: FnInit,
    free: FnVoidPtr,
    model_free: FnVoidPtr,
    model_get_vocab: FnPtrRet,
    vocab_n_tokens: FnI32,
    backend_init: FnVoid,
    batch_one: FnBatch,
    decode: FnDecode,
    logits_ith: FnLogits,
    tokenize: FnTokenize,
    detok: FnDetok,
    memory_clear: unsafe extern "C" fn(*mut c_void, bool),
    get_memory: FnPtrRet,
    n_ctx: FnI32,
}

impl Lib {
    fn open() -> Result<Self> {
        let dir = std::env::var("CORTEX_GGML_DIR")
            .unwrap_or_else(|_| "/home/serjey/.unsloth/llama.cpp/build/bin".into());
        // Preload the dependencies by absolute path so dlopen can resolve
        // their SONAMEs without LD_LIBRARY_PATH being set at process start.
        let cudart_dir = std::env::var("CORTEX_CUDART_DIR").unwrap_or_else(|_| {
            "/home/serjey/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/nvidia/cu13/lib".into()
        });
        let mut paths: Vec<String> = Vec::new();
        for name in ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"] {
            paths.push(format!("{cudart_dir}/{name}"));
        }
        for name in [
            "libggml-base.so.1",
            "libggml.so.1",
            "libggml-cuda.so",
            "libllama.so.1",
        ] {
            paths.push(format!("{dir}/{name}"));
        }
        for path in paths {
            let name = path.rsplit('/').next().unwrap_or(&path);
            let c = CString::new(path.clone()).unwrap();
            let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if h.is_null() {
                let e = unsafe { dlerror() };
                let msg = if e.is_null() {
                    "unknown".into()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(e) }.to_string_lossy().into_owned()
                };
                return Err(Error::InvalidConfig(format!("cannot load {path}: {msg}")));
            }
        }
        let c = CString::new(format!("{dir}/libllama.so.1")).unwrap();
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(Error::InvalidConfig("cannot load libllama.so.1".into()));
        }
        let c = CString::new(format!("{dir}/libggml.so.1")).unwrap();
        let ggml = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if ggml.is_null() {
            return Err(Error::InvalidConfig("cannot load libggml.so.1".into()));
        }
        unsafe {
            let ggml_log_set: unsafe extern "C" fn(
                Option<unsafe extern "C" fn(c_int, *const c_char, *mut c_void)>, *mut c_void,
            ) = sym(ggml, "ggml_log_set")?;
            unsafe extern "C" fn quiet(_: c_int, _: *const c_char, _: *mut c_void) {}
            ggml_log_set(Some(quiet), std::ptr::null_mut());
            let load_all: unsafe extern "C" fn(*const c_char) = sym(ggml, "ggml_backend_load_all_from_path")?;
            let d = CString::new(dir.clone()).unwrap();
            load_all(d.as_ptr());
            let backend_init: FnVoid = sym(h, "llama_backend_init")?;
            backend_init();
            // Silence llama.cpp's verbose graph dumps; they cost more than
            // the decode itself at batch=1.
            let log_set: unsafe extern "C" fn(Option<unsafe extern "C" fn(/*level*/ c_int, *const c_char, *mut c_void)>, *mut c_void) =
                sym(h, "llama_log_set")?;
            log_set(Some(quiet), std::ptr::null_mut());
            Ok(Self {
                _h: h,
                _ggml: ggml,
                default_model: sym(h, "llama_model_default_params")?,
                default_ctx: sym(h, "llama_context_default_params")?,
                load: sym(h, "llama_model_load_from_file")?,
                init: sym(h, "llama_init_from_model")?,
                free: sym(h, "llama_free")?,
                model_free: sym(h, "llama_model_free")?,
                model_get_vocab: sym(h, "llama_model_get_vocab")?,
                vocab_n_tokens: sym(h, "llama_vocab_n_tokens")?,
                backend_init,
                batch_one: sym(h, "llama_batch_get_one")?,
                decode: sym(h, "llama_decode")?,
                logits_ith: sym(h, "llama_get_logits_ith")?,
                tokenize: sym(h, "llama_tokenize")?,
                detok: sym(h, "llama_detokenize")?,
                memory_clear: sym(h, "llama_memory_clear")?,
                get_memory: sym(h, "llama_get_memory")?,
                n_ctx: sym(h, "llama_n_ctx")?,
            })
        }
    }
}

/// Loaded llama.cpp model with all layers offloaded to CUDA.
pub struct GgmlCuda {
    lib: Lib,
    model: *mut c_void,
    ctx: *mut c_void,
    vocab: *mut c_void,
    pub vocab_size: usize,
    pub context: usize,
}

fn put_i32(buf: &mut [u8], off: usize, v: i32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

impl GgmlCuda {
    pub fn load(path: impl AsRef<Path>, context: usize, threads: i32) -> Result<Self> {
        let lib = Lib::open()?;
        let mut mp = unsafe { (lib.default_model)() };
        // n_gpu_layers = -1 at offset 16 (see mirrored header layout).
        put_i32(&mut mp.0, 16, -1);
        let path = CString::new(path.as_ref().to_string_lossy().as_bytes()).unwrap();
        let model = unsafe { (lib.load)(path.as_ptr(), mp) };
        if model.is_null() {
            return Err(Error::InvalidConfig("llama_model_load_from_file failed".into()));
        }
        let mut cp = unsafe { (lib.default_ctx)() };
        put_i32(&mut cp.0, 0, context as i32); // n_ctx
        put_i32(&mut cp.0, 4, 512); // n_batch
        put_i32(&mut cp.0, 8, 512); // n_ubatch
        put_i32(&mut cp.0, 28, threads); // n_threads
        put_i32(&mut cp.0, 32, threads); // n_threads_batch
        let ctx = unsafe { (lib.init)(model, cp) };
        if ctx.is_null() {
            unsafe { (lib.model_free)(model) };
            return Err(Error::InvalidConfig("llama_init_from_model failed".into()));
        }
        let vocab = unsafe { (lib.model_get_vocab)(model) };
        let vocab_size = unsafe { (lib.vocab_n_tokens)(vocab) }.max(0) as usize;
        let got_ctx = unsafe { (lib.n_ctx)(ctx) }.max(0) as usize;
        Ok(Self {
            lib,
            model,
            ctx,
            vocab,
            vocab_size,
            context: got_ctx,
        })
    }

    pub fn reset(&mut self) {
        unsafe {
            let mem = (self.lib.get_memory)(self.ctx);
            if !mem.is_null() {
                (self.lib.memory_clear)(mem, true);
            }
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let c = CString::new(text).unwrap();
        let mut out = vec![0i32; text.len() + 8];
        let n = unsafe {
            (self.lib.tokenize)(
                self.vocab,
                c.as_ptr(),
                text.len() as c_int,
                out.as_mut_ptr(),
                out.len() as c_int,
                false,
                true,
            )
        };
        if n < 0 {
            return Err(Error::InvalidConfig("llama_tokenize failed".into()));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    pub fn decode_text(&self, tokens: &[i32]) -> Result<String> {
        let mut buf = vec![0i8; tokens.len() * 8 + 16];
        let n = unsafe {
            (self.lib.detok)(
                self.vocab,
                tokens.as_ptr(),
                tokens.len() as c_int,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                false,
            )
        };
        if n < 0 {
            return Err(Error::InvalidConfig("llama_detokenize failed".into()));
        }
        let bytes: Vec<u8> = buf[..n as usize].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Greedy generation. Prompt tokens are prefilled in one batch.
    pub fn generate_greedy(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        self.reset();
        let mut ids = self.encode(prompt)?;
        if ids.is_empty() {
            return Err(Error::InvalidConfig("empty prompt".into()));
        }
        let n_prompt = ids.len();
        let batch = unsafe { (self.lib.batch_one)(ids.as_mut_ptr(), n_prompt as c_int) };
        let rc = unsafe { (self.lib.decode)(self.ctx, batch) };
        if rc != 0 {
            return Err(Error::InvalidConfig(format!("llama_decode(prefill)={rc}")));
        }
        let p = unsafe { (self.lib.logits_ith)(self.ctx, n_prompt as c_int - 1) };
        if p.is_null() {
            return Err(Error::InvalidConfig("null logits after prefill".into()));
        }
        let mut next = Self::argmax(unsafe { std::slice::from_raw_parts(p, self.vocab_size) });
        let mut out = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            out.push(next);
            let mut one = [next as i32];
            let batch = unsafe { (self.lib.batch_one)(one.as_mut_ptr(), 1) };
            let rc = unsafe { (self.lib.decode)(self.ctx, batch) };
            if rc != 0 {
                return Err(Error::InvalidConfig(format!("llama_decode={rc}")));
            }
            let p = unsafe { (self.lib.logits_ith)(self.ctx, 0) };
            next = Self::argmax(unsafe { std::slice::from_raw_parts(p, self.vocab_size) });
        }
        Ok(out)
    }

    /// Decode-only throughput: prefill once, then time `n` greedy forwards.
    pub fn bench(&mut self, prompt: &str, n: usize) -> Result<(usize, f64)> {
        self.reset();
        let mut ids = self.encode(prompt)?;
        let n_prompt = ids.len();
        let batch = unsafe { (self.lib.batch_one)(ids.as_mut_ptr(), n_prompt as c_int) };
        let rc = unsafe { (self.lib.decode)(self.ctx, batch) };
        if rc != 0 {
            return Err(Error::InvalidConfig(format!("llama_decode(prefill)={rc}")));
        }
        let p = unsafe { (self.lib.logits_ith)(self.ctx, n_prompt as c_int - 1) };
        let mut next = Self::argmax(unsafe { std::slice::from_raw_parts(p, self.vocab_size) });
        let t = Instant::now();
        for _ in 0..n {
            let mut one = [next as i32];
            let batch = unsafe { (self.lib.batch_one)(one.as_mut_ptr(), 1) };
            let rc = unsafe { (self.lib.decode)(self.ctx, batch) };
            if rc != 0 {
                return Err(Error::InvalidConfig(format!("llama_decode={rc}")));
            }
            let p = unsafe { (self.lib.logits_ith)(self.ctx, 0) };
            next = Self::argmax(unsafe { std::slice::from_raw_parts(p, self.vocab_size) });
        }
        Ok((n, t.elapsed().as_secs_f64()))
    }
}

impl Drop for GgmlCuda {
    fn drop(&mut self) {
        unsafe {
            (self.lib.free)(self.ctx);
            (self.lib.model_free)(self.model);
        }
    }
}
