//! Minimal 2-D tensor type and matrix-vector product.
//!
//! No BLAS, no ndarray: just contiguous `Vec<f32>` and a cache-friendly dot
//! product.  Quantized matrices can either be dequantized eagerly at load
//! time (fast, memory hungry) or kept compressed and dequantized one row at
//! a time on every use (small memory footprint).

use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::gguf::Mmap;
use crate::thread_pool::ThreadPool;
use crate::quant::{
    dequantize_row_layout, detect_q8_1_layout, ggml_type, is_supported, row_bytes, type_name,
    Q8Layout,
};

/// Whether a GGUF tensor is expanded to `f32` when the model is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DequantPolicy {
    /// Dequantize everything once, at load time. Fast inference, ~4 bytes/weight.
    Eager,
    /// Keep the original bytes; dequantize a row whenever it is used.
    /// This is the default and keeps memory close to the file size.
    Lazy,
}

enum Bytes {
    Owned(Arc<Vec<u8>>),
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v.as_slice(),
            Bytes::Mapped { map, offset, len } => map
                .slice(*offset, *len)
                .expect("tensor range was validated when the view was created"),
        }
    }
}

enum Storage {
    F32(Vec<f32>),
    Raw {
        kind: u32,
        bytes: Bytes,
        q8_layout: Q8Layout,
    },
}

/// A row-major matrix with `rows` output features and `cols` input features.
pub struct Matrix {
    rows: usize,
    cols: usize,
    kind: u32,
    storage: Storage,
}

impl Matrix {
    /// Build a matrix from owned `f32` data.
    pub fn from_f32(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows.saturating_mul(cols) {
            return Err(Error::ShapeMismatch(format!(
                "expected {} f32 values, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            kind: ggml_type::F32,
            storage: Storage::F32(data),
        })
    }

    /// Wrap a 1-D vector as a single-row matrix.
    pub fn from_vector(data: Vec<f32>) -> Self {
        let cols = data.len();
        Self {
            rows: 1,
            cols,
            kind: ggml_type::F32,
            storage: Storage::F32(data),
        }
    }

    /// Build a matrix from GGUF tensor bytes (copies the bytes).
    pub fn from_gguf(
        kind: u32,
        rows: usize,
        cols: usize,
        bytes: &[u8],
        policy: DequantPolicy,
    ) -> Result<Self> {
        Self::from_bytes(
            kind,
            rows,
            cols,
            Bytes::Owned(Arc::new(bytes.to_vec())),
            policy,
        )
    }

    /// Build a matrix from a zero-copy slice of a memory-mapped GGUF file.
    pub fn from_gguf_mapped(
        kind: u32,
        rows: usize,
        cols: usize,
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
        policy: DequantPolicy,
    ) -> Result<Self> {
        Self::from_bytes(
            kind,
            rows,
            cols,
            Bytes::Mapped { map, offset, len },
            policy,
        )
    }

    fn from_bytes(
        kind: u32,
        rows: usize,
        cols: usize,
        bytes: Bytes,
        policy: DequantPolicy,
    ) -> Result<Self> {
        if !is_supported(kind) {
            return Err(Error::UnsupportedType(kind));
        }
        let needed = row_bytes(kind, cols).saturating_mul(rows);
        if bytes.as_slice().len() < needed {
            return Err(Error::ShapeMismatch(format!(
                "{} matrix {rows}x{cols} needs {needed} bytes, got {}",
                type_name(kind),
                bytes.as_slice().len()
            )));
        }

        // Q8_1 changed its field order between ggml revisions; decide once
        // per tensor so every row is decoded consistently.
        let q8_layout = if kind == ggml_type::Q8_1 {
            detect_q8_1_layout(bytes.as_slice(), cols, rows)
        } else {
            Q8Layout::New
        };

        let storage = match policy {
            DequantPolicy::Eager => {
                let mut values = vec![0.0f32; rows * cols];
                let rb = row_bytes(kind, cols);
                let data = bytes.as_slice();
                for r in 0..rows {
                    let src = &data[r * rb..(r + 1) * rb];
                    let dst = &mut values[r * cols..(r + 1) * cols];
                    if !dequantize_row_layout(kind, src, dst, cols, q8_layout) {
                        return Err(Error::UnsupportedType(kind));
                    }
                }
                Storage::F32(values)
            }
            DequantPolicy::Lazy => Storage::Raw { kind, bytes, q8_layout },
        };

        Ok(Self {
            rows,
            cols,
            kind,
            storage,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn kind(&self) -> u32 {
        self.kind
    }

    /// Raw compressed bytes for lazily stored matrices (no copy).
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        match &self.storage {
            Storage::Raw { .. } => Some(self.as_raw_bytes()),
            Storage::F32(_) => None,
        }
    }

    fn as_raw_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Raw { bytes, .. } => bytes.as_slice(),
            _ => unreachable!(),
        }
    }

    /// Returns the expanded weights if the matrix is stored as `f32`.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::F32(v) => Some(v),
            Storage::Raw { .. } => None,
        }
    }

    /// Copy one row into `out`, dequantizing on the fly if needed.
    pub fn row(&self, r: usize, out: &mut [f32]) -> Result<()> {
        if r >= self.rows || out.len() != self.cols {
            return Err(Error::ShapeMismatch(format!(
                "row {r} of {}x{} matrix does not fit {} slots",
                self.rows,
                self.cols,
                out.len()
            )));
        }
        match &self.storage {
            Storage::F32(v) => {
                out.copy_from_slice(&v[r * self.cols..(r + 1) * self.cols]);
            }
            Storage::Raw {
                kind,
                bytes,
                q8_layout,
            } => {
                let rb = row_bytes(*kind, self.cols);
                let mut tmp = vec![0.0f32; self.cols];
                if !dequantize_row_layout(
                    *kind,
                    &bytes.as_slice()[r * rb..(r + 1) * rb],
                    &mut tmp,
                    self.cols,
                    *q8_layout,
                ) {
                    return Err(Error::UnsupportedType(*kind));
                }
                out.copy_from_slice(&tmp);
            }
        }
        Ok(())
    }

    /// Compute `out = self * x`, i.e. one dot product per row.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.check_shapes(x, out)?;
        self.matvec_range(x, out, 0, self.rows)
    }

    /// Parallel version: rows are split into contiguous chunks, one per thread.
    pub fn matvec_parallel(&self, x: &[f32], out: &mut [f32], threads: usize) -> Result<()> {
        self.check_shapes(x, out)?;
        if threads <= 1 || self.rows < 256 {
            return self.matvec_range(x, out, 0, self.rows);
        }
        let threads = threads.min(self.rows);
        let chunk = self.rows.div_ceil(threads);
        let errors = std::sync::Mutex::new(Vec::new());
        struct SendPtr(*mut f32);
        // SAFETY: each thread receives a distinct, non-overlapping slice of
        // `out`, and `thread::scope` joins all workers before returning.
        unsafe impl Send for SendPtr {}
        let out_ptr = SendPtr(out.as_mut_ptr());
        std::thread::scope(|scope| {
            for t in 0..threads {
                let start = t * chunk;
                let end = (start + chunk).min(self.rows);
                if start >= end {
                    break;
                }
                let len = end - start;
                let ptr = unsafe { out_ptr.0.add(start) } as usize;
                let errors = &errors;
                scope.spawn(move || {
                    let chunk = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
                    if let Err(e) = self.matvec_range(x, chunk, start, end) {
                        errors.lock().unwrap().push(e);
                    }
                });
            }
        });
        let mut errors = errors.into_inner().unwrap();
        if let Some(e) = errors.pop() {
            return Err(e);
        }
        Ok(())
    }

    /// Matrix-vector product using the persistent worker pool.
    pub fn matvec_pool(&self, x: &[f32], out: &mut [f32], pool: &ThreadPool) -> Result<()> {
        self.check_shapes(x, out)?;
        if pool.len() <= 1 || self.rows < 256 {
            return self.matvec_range(x, out, 0, self.rows);
        }
        let tasks = pool.len().min(self.rows);
        let chunk = self.rows.div_ceil(tasks);
        let m_ptr = self as *const Matrix as usize;
        let x_ptr = x.as_ptr() as usize;
        let out_ptr = out.as_mut_ptr() as usize;
        let rows = self.rows;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let worker_errors = Arc::clone(&errors);
        pool.dispatch(tasks, move |t| {
            let start = t * chunk;
            let end = (start + chunk).min(rows);
            if start >= end {
                return;
            }
            let len = end - start;
            // SAFETY: the pool's barrier keeps these references alive until
            // all jobs have completed; row ranges do not overlap.
            let m = unsafe { &*(m_ptr as *const Matrix) };
            let x = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, m.cols) };
            let o = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), len) };
            if let Err(e) = m.matvec_range(x, o, start, end) {
                worker_errors.lock().unwrap().push(e);
            }
        });
        let mut errors = errors.lock().unwrap();
        if let Some(e) = errors.pop() {
            return Err(e);
        }
        Ok(())
    }

    fn check_shapes(&self, x: &[f32], out: &[f32]) -> Result<()> {
        if x.len() != self.cols || out.len() != self.rows {
            return Err(Error::ShapeMismatch(format!(
                "matvec: matrix {}x{}, x.len={}, out.len={}",
                self.rows,
                self.cols,
                x.len(),
                out.len()
            )));
        }
        Ok(())
    }

    fn matvec_range(&self, x: &[f32], out: &mut [f32], start: usize, end: usize) -> Result<()> {
        match &self.storage {
            Storage::F32(v) => {
                for r in start..end {
                    out[r - start] = dot(&v[r * self.cols..(r + 1) * self.cols], x);
                }
            }
            Storage::Raw {
                kind,
                bytes,
                q8_layout,
            } => {
                let rb = row_bytes(*kind, self.cols);
                let bb = crate::quant::block_bytes(*kind);
                let be = crate::quant::block_elements(*kind).max(1);
                let mut tmp = vec![0.0f32; self.cols.max(be)];
                // Dequantize one small block at a time into a cache-resident
                // buffer and accumulate immediately. This avoids materializing
                // a full f32 row in memory for every matvec.
                for r in start..end {
                    let row = &bytes.as_slice()[r * rb..(r + 1) * rb];
                    if let Some(v) = crate::quant::dot_quant(*kind, row, x, self.cols, *q8_layout) {
                        out[r - start] = v;
                    } else if *kind == ggml_type::F32 || *kind == ggml_type::F16
                        || *kind == ggml_type::BF16
                    {
                        if !dequantize_row_layout(*kind, row, &mut tmp, self.cols, *q8_layout) {
                            return Err(Error::UnsupportedType(*kind));
                        }
                        out[r - start] = dot(&tmp, x);
                    } else {
                        let mut sum = 0.0f32;
                        let mut c = 0usize;
                        let mut bo = 0usize;
                        while c < self.cols {
                            let n = be.min(self.cols - c);
                            if !dequantize_row_layout(
                                *kind,
                                &row[bo..bo + bb],
                                &mut tmp[..n],
                                n,
                                *q8_layout,
                            ) {
                                return Err(Error::UnsupportedType(*kind));
                            }
                            sum += dot(&tmp[..n], &x[c..c + n]);
                            c += be;
                            bo += bb;
                        }
                        out[r - start] = sum;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Dot product, dispatching to AVX2+FMA when the CPU supports it.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: feature detection above guarantees AVX support.
            return unsafe { dot_avx(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    while i + 16 <= n {
        acc0 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i))),
            acc0,
        );
        acc1 = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            ),
            acc1,
        );
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = lanes.iter().sum::<f32>();
    while i < n {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Portable fallback with four independent accumulators.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_f32() {
        let m = Matrix::from_f32(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut out = [0.0; 2];
        m.matvec(&[1.0, 1.0, 1.0], &mut out).unwrap();
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn eager_and_lazy_agree() {
        // F16 values 1.0, -2.0, 0.5, 3.0 as a 2x2 matrix
        let mut bytes = Vec::new();
        for h in [0x3c00u16, 0xc000, 0x3800, 0x4200] {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        let eager = Matrix::from_gguf(ggml_type::F16, 2, 2, &bytes, DequantPolicy::Eager).unwrap();
        let lazy = Matrix::from_gguf(ggml_type::F16, 2, 2, &bytes, DequantPolicy::Lazy).unwrap();
        let x = [0.25, 4.0];
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        eager.matvec(&x, &mut a).unwrap();
        lazy.matvec(&x, &mut b).unwrap();
        assert_eq!(a, b);
    }
}
