//! Smoke test / benchmark for the CUDA matvec backend.
//!
//!   cargo run --release --example cuda_bench

use std::time::Instant;

use cortex::cuda::CudaDevice;
use cortex::tensor::{dot, Matrix};

fn main() -> Result<(), cortex::Error> {
    let device = CudaDevice::new()?;
    println!("CUDA device initialized, kernel JIT-compiled");

    let rows = 4096;
    let cols = 4096;
    let mut s = 1u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let x: Vec<f32> = (0..cols).map(|_| next()).collect();
    let w_gpu = device.upload(&w)?;

    let mut gpu_out = vec![0.0f32; rows];
    let t0 = Instant::now();
    device.matvec(&w_gpu, &x, &mut gpu_out)?;
    println!("GPU matvec: {:?}", t0.elapsed());

    let cpu = Matrix::from_f32(rows, cols, w.clone())?;
    let mut cpu_out = vec![0.0f32; rows];
    let t1 = Instant::now();
    cpu.matvec(&x, &mut cpu_out)?;
    println!("CPU matvec: {:?}", t1.elapsed());

    let mut max_err = 0f32;
    for r in 0..rows {
        let e = (gpu_out[r] - cpu_out[r]).abs();
        max_err = max_err.max(e);
    }
    let expected = dot(&w[..cols], &x);
    println!("max |gpu-cpu| = {max_err:.3e}, first row expected {expected:.4} got {:.4}", gpu_out[0]);
    assert!(max_err < 1e-2, "GPU result differs from CPU");
    Ok(())
}
