use blstrs::{Bls12, G1Affine, G1Projective, G2Affine, G2Projective, Scalar};
use group::{Curve, Group};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use webgpu_groth16::gpu::GpuContext;
use webgpu_groth16::prover;

#[derive(Serialize)]
struct BenchRow {
    size: usize,
    init_ms: f64,
    prep_ms: f64,
    total_ms: f64,
    total_with_init_ms: f64,
}

#[derive(Serialize)]
struct BenchReport {
    init_ms: f64,
    rows: Vec<BenchRow>,
}

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

fn now_ms() -> Result<f64, JsValue> {
    let window =
        web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let performance = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;
    Ok(performance.now())
}

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

fn build_dataset_g2(n: usize) -> (Vec<G2Affine>, Vec<Scalar>) {
    let generator = G2Projective::generator();
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

#[wasm_bindgen]
pub async fn benchmark_msm(
    min_log2: u32,
    max_log2: u32,
    iterations: u32,
) -> Result<JsValue, JsValue> {
    install_panic_hook();
    let t_init = now_ms()?;
    log("heliax: GpuContext::new() start");
    let gpu = GpuContext::<Bls12>::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu init failed: {e:#}")))?;
    log("heliax: GpuContext::new() done");
    let init_ms = now_ms()? - t_init;

    let mut rows = Vec::new();
    for log2 in min_log2..=max_log2 {
        let size = 1usize << log2;

        let t_prep = now_ms()?;
        let (bases, scalars) = build_dataset(size);
        let prep_ms = now_ms()? - t_prep;
        log(&format!("heliax: dataset built n={size}"));

        let mut total_ms_acc = 0.0;
        for _ in 0..iterations.max(1) {
            let t_total = now_ms()?;
            log(&format!("heliax: gpu_msm_g1 start n={size}"));
            let _result = prover::gpu_msm_g1::<Bls12>(&gpu, &bases, &scalars)
                .await
                .map_err(|e| JsValue::from_str(&format!("gpu_msm_g1 failed: {e:#}")))?;
            log(&format!("heliax: gpu_msm_g1 done n={size}"));
            total_ms_acc += now_ms()? - t_total;
        }

        let total_ms = total_ms_acc / iterations.max(1) as f64;
        rows.push(BenchRow {
            size,
            init_ms,
            prep_ms,
            total_ms,
            total_with_init_ms: total_ms + init_ms,
        });
    }

    serde_wasm_bindgen::to_value(&BenchReport { init_ms, rows })
        .map_err(|e| JsValue::from_str(&format!("serialize report failed: {e}")))
}

#[wasm_bindgen]
pub async fn benchmark_msm_g2(
    min_log2: u32,
    max_log2: u32,
    iterations: u32,
) -> Result<JsValue, JsValue> {
    install_panic_hook();
    let t_init = now_ms()?;
    log("heliax: GpuContext::new() start");
    let gpu = GpuContext::<Bls12>::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu init failed: {e:#}")))?;
    log("heliax: GpuContext::new() done");
    let init_ms = now_ms()? - t_init;

    let empty_g1_bases: Vec<G1Affine> = Vec::new();
    let empty_scalars: Vec<Scalar> = Vec::new();

    let mut rows = Vec::new();
    for log2 in min_log2..=max_log2 {
        let size = 1usize << log2;

        let t_prep = now_ms()?;
        let (bases, scalars) = build_dataset_g2(size);
        let prep_ms = now_ms()? - t_prep;
        log(&format!("heliax: g2 dataset built n={size}"));

        let mut total_ms_acc = 0.0;
        for _ in 0..iterations.max(1) {
            let t_total = now_ms()?;
            log(&format!("heliax: gpu_msm_g2(batch) start n={size}"));
            let _result = prover::gpu_msm_batch::<Bls12>(
                &gpu,
                &empty_g1_bases,
                &empty_scalars,
                &empty_g1_bases,
                &empty_scalars,
                &empty_g1_bases,
                &empty_scalars,
                &empty_g1_bases,
                &empty_scalars,
                &bases,
                &scalars,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("gpu_msm_g2(batch) failed: {e:#}")))?;
            log(&format!("heliax: gpu_msm_g2(batch) done n={size}"));
            total_ms_acc += now_ms()? - t_total;
        }

        let total_ms = total_ms_acc / iterations.max(1) as f64;
        rows.push(BenchRow {
            size,
            init_ms,
            prep_ms,
            total_ms,
            total_with_init_ms: total_ms + init_ms,
        });
    }

    serde_wasm_bindgen::to_value(&BenchReport { init_ms, rows })
        .map_err(|e| JsValue::from_str(&format!("serialize report failed: {e}")))
}

#[wasm_bindgen]
pub async fn probe_init() -> Result<JsValue, JsValue> {
    install_panic_hook();
    log("heliax: probe_init start");
    let t = now_ms()?;
    let _gpu = GpuContext::<Bls12>::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu init failed: {e:#}")))?;
    log("heliax: probe_init done");
    Ok(JsValue::from_f64(now_ms()? - t))
}

#[wasm_bindgen]
pub async fn probe_msm_once(size: u32) -> Result<JsValue, JsValue> {
    install_panic_hook();
    log("heliax: probe_msm_once init start");
    let gpu = GpuContext::<Bls12>::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu init failed: {e:#}")))?;
    log("heliax: probe_msm_once init done");
    let (bases, scalars) = build_dataset(size as usize);
    log(&format!("heliax: probe_msm_once dataset n={size}"));
    let t = now_ms()?;
    let _result = prover::gpu_msm_g1::<Bls12>(&gpu, &bases, &scalars)
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu_msm_g1 failed: {e:#}")))?;
    log(&format!("heliax: probe_msm_once msm done n={size}"));
    Ok(JsValue::from_f64(now_ms()? - t))
}

#[wasm_bindgen]
pub async fn probe_msm_g2_once(size: u32) -> Result<JsValue, JsValue> {
    install_panic_hook();
    log("heliax: probe_msm_g2_once init start");
    let gpu = GpuContext::<Bls12>::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("gpu init failed: {e:#}")))?;
    log("heliax: probe_msm_g2_once init done");
    let (bases, scalars) = build_dataset_g2(size as usize);
    let empty_g1_bases: Vec<G1Affine> = Vec::new();
    let empty_scalars: Vec<Scalar> = Vec::new();
    log(&format!("heliax: probe_msm_g2_once dataset n={size}"));
    let t = now_ms()?;
    let _result = prover::gpu_msm_batch::<Bls12>(
        &gpu,
        &empty_g1_bases,
        &empty_scalars,
        &empty_g1_bases,
        &empty_scalars,
        &empty_g1_bases,
        &empty_scalars,
        &empty_g1_bases,
        &empty_scalars,
        &bases,
        &scalars,
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("gpu_msm_g2(batch) failed: {e:#}")))?;
    log(&format!("heliax: probe_msm_g2_once msm done n={size}"));
    Ok(JsValue::from_f64(now_ms()? - t))
}
