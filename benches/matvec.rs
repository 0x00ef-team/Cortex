//! Micro-benchmark for the matrix-vector kernel (`cargo bench`).

use std::hint::black_box;
use std::time::Instant;

use cortex::quant::ggml_type;
use cortex::tensor::{DequantPolicy, Matrix};

fn run(name: &str, m: &Matrix, x: &[f32], iterations: usize) {
    let mut out = vec![0.0f32; m.rows()];
    // warm-up
    for _ in 0..16 {
        m.matvec(x, &mut out).unwrap();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        m.matvec(black_box(x), black_box(&mut out)).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let flops = 2.0 * m.rows() as f64 * m.cols() as f64 * iterations as f64;
    println!(
        "{name:<12} {:>7.2} ms/iter  {:>7.2} GFLOP/s  ({}x{})",
        elapsed * 1000.0 / iterations as f64,
        flops / elapsed / 1e9,
        m.rows(),
        m.cols()
    );
}

fn main() {
    let rows = 4096;
    let cols = 4096;
    let iterations = 20;

    let mut data = Vec::with_capacity(rows * cols);
    let mut s = 1u64;
    for _ in 0..rows * cols {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push(((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0);
    }
    let x: Vec<f32> = (0..cols).map(|i| i as f32 / cols as f32).collect();

    let dense = Matrix::from_f32(rows, cols, data.clone()).unwrap();
    run("F32", &dense, &x, iterations);

    // Same values as F16 bytes for the lazy dequantization path.
    let mut f16 = Vec::with_capacity(rows * cols * 2);
    for v in &data {
        f16.extend_from_slice(&to_half(*v).to_le_bytes());
    }
    let lazy = Matrix::from_gguf(ggml_type::F16, rows, cols, &f16, DequantPolicy::Lazy).unwrap();
    run("F16 lazy", &lazy, &x, iterations);

    // Q5_0 row bytes: 22 per 32 elements.
    let q5_rows = 4096;
    let q5_cols = 4096;
    let mut q5 = Vec::with_capacity(q5_rows * q5_cols / 32 * 22);
    for _ in 0..q5_rows * q5_cols / 32 {
        q5.extend_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        q5.extend_from_slice(&1234u32.to_le_bytes());   // qh
        for _ in 0..16 {
            q5.push(0x78);
        }
    }
    let q5m = Matrix::from_gguf(ggml_type::Q5_0, q5_rows, q5_cols, &q5, DequantPolicy::Lazy).unwrap();
    run("Q5_0 fused", &q5m, &x, iterations);

    let q5e = Matrix::from_gguf(ggml_type::Q5_0, q5_rows, q5_cols, &q5, DequantPolicy::Eager).unwrap();
    run("Q5_0 eager", &q5e, &x, iterations);

    for (kind, block, name) in [
        (ggml_type::Q4_K, 144usize, "Q4_K fused"),
        (ggml_type::Q6_K, 210usize, "Q6_K fused"),
    ] {
        let r = 4096;
        let c = 4096;
        let blocks = r * c / 256;
        let mut raw = vec![0u8; blocks * block];
        let mut s = 99u64;
        for b in &mut raw {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (s >> 33) as u8;
        }
        for off in (0..raw.len()).step_by(block) {
            let at = if kind == ggml_type::Q4_K { off } else { off + 208 };
            raw[at..at + 2].copy_from_slice(&0x3800u16.to_le_bytes());
            if kind == ggml_type::Q4_K {
                raw[off + 2..off + 4].copy_from_slice(&0x3000u16.to_le_bytes());
            }
        }
        let m = Matrix::from_gguf(kind, r, c, &raw, DequantPolicy::Lazy).unwrap();
        run(name, &m, &x, iterations / 2);
    }
}

fn to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32 - 127 + 15) as u16;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    if exp == 0 || exp >= 31 {
        // The benchmark data is small; saturate rather than implementing
        // full round-to-nearest for edge cases.
        return sign | 0x7bff;
    }
    sign | (exp << 10) | mant
}
