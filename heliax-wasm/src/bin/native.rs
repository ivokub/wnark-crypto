use std::time::Instant;

use blstrs::{Bls12, G1Affine, G1Projective, Scalar};
use group::{Curve, Group};
use webgpu_groth16::gpu::GpuContext;
use webgpu_groth16::prover;

fn scalar_for_index(i: usize) -> Scalar {
    let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    Scalar::from(x ^ (x >> 17) ^ 0xdead_beef_u64)
}

fn build_dataset(n: usize) -> (Vec<G1Affine>, Vec<Scalar>) {
    let generator = G1Projective::generator();
    let mut current = generator;
    let mut bases = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    for i in 0..n {
        bases.push(current.to_affine());
        scalars.push(scalar_for_index(i));
        current += generator;
    }
    (bases, scalars)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let min_log2: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let max_log2: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(19);
    let iterations: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    eprintln!("=== Heliax MSM Native Benchmark ===");
    let t_init = Instant::now();
    let gpu = pollster::block_on(GpuContext::<Bls12>::new())
        .expect("GpuContext::new failed");
    let init_ms = t_init.elapsed().as_secs_f64() * 1000.0;
    eprintln!("init_ms = {:.3}", init_ms);
    eprintln!();
    println!("size,init_ms,prep_ms,total_ms,total_with_init_ms");

    for log2 in min_log2..=max_log2 {
        let size = 1usize << log2;
        let t_prep = Instant::now();
        let (bases, scalars) = build_dataset(size);
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;

        let mut total_ms_acc = 0.0;
        for _ in 0..iterations.max(1) {
            let t_total = Instant::now();
            let _result = pollster::block_on(prover::gpu_msm_g1::<Bls12>(
                &gpu,
                &bases,
                &scalars,
            ))
            .expect("gpu_msm_g1 failed");
            total_ms_acc += t_total.elapsed().as_secs_f64() * 1000.0;
        }
        let total_ms = total_ms_acc / iterations.max(1) as f64;
        println!(
            "{},{:.3},{:.3},{:.3},{:.3}",
            size,
            init_ms,
            prep_ms,
            total_ms,
            total_ms + init_ms
        );
    }
}
