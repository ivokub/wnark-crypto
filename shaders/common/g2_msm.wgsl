var<workgroup> g2_window_shared: array<G2Point, 32>;

fn g2_add_affine(a: G2Point, b: G2Point) -> G2Point {
  return g2_jac_to_affine(g2_add_mixed(g2_affine_to_jac(a), b));
}

fn msm_params_terms_per_instance() -> u32 {
  return params.lane0.z;
}

fn msm_params_window() -> u32 {
  return params.lane0.w;
}

fn msm_params_num_windows() -> u32 {
  return params.lane1.x;
}

fn g2_scalar_mul_affine_small(base: G2Point, scalar: u32) -> G2Point {
  if (scalar == 0u || g2_affine_is_infinity(base)) {
    return g2_jac_to_affine(g2_jac_infinity());
  }
  var acc = g2_jac_infinity();
  var k = scalar;
  loop {
    if (k == 0u) {
      break;
    }
    acc = g2_add_mixed(acc, base);
    k = k - 1u;
  }
  return g2_jac_to_affine(acc);
}

@compute @workgroup_size(32)
fn g2_msm_bucket_sparse_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let start = input_meta1[i];
  let size = input_meta2[i];
  var acc = g2_jac_infinity();
  for (var j = 0u; j < size; j = j + 1u) {
    let raw = input_meta0[start + j];
    let idx = raw & 0x7fffffffu;
    let neg = (raw & 0x80000000u) != 0u;
    var point = g2_load_from(0u, idx);
    if (neg) {
      point = g2_neg_affine(point);
    }
    acc = g2_add_mixed(acc, point);
  }
  g2_store(i, g2_jac_to_affine(acc));
}

@compute @workgroup_size(64)
fn g2_msm_weight_buckets_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let point = g2_load_from(0u, i);
  let value = input_meta0[i];
  if (g2_affine_is_infinity(point) || value == 0u) {
    g2_store(i, g2_jac_to_affine(g2_jac_infinity()));
    return;
  }
  g2_store(i, g2_scalar_mul_affine_small(point, value));
}

@compute @workgroup_size(32)
fn g2_msm_subsum_phase1_main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
) {
  let i = wg_id.x;
  let tid = local_id.x;
  if (i >= params_count()) {
    return;
  }
  let start = input_meta1[i];
  let count = input_meta2[i];
  var local_sum = g2_jac_infinity();
  var j = tid;
  loop {
    if (j >= count) {
      break;
    }
    let point = g2_load_from(0u, start + j);
    if (!g2_affine_is_infinity(point)) {
      local_sum = g2_add_mixed(local_sum, point);
    }
    j = j + 32u;
  }
  g2_window_shared[tid] = g2_jac_to_affine(local_sum);
  workgroupBarrier();

  var stride = 16u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      g2_window_shared[tid] = g2_add_affine(g2_window_shared[tid], g2_window_shared[tid + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    g2_store(i, g2_window_shared[0u]);
  }
}

@compute @workgroup_size(64)
fn g2_msm_combine_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let num_windows = msm_params_num_windows();
  let window = msm_params_window();
  var acc = g2_jac_infinity();
  for (var win: i32 = i32(num_windows) - 1; win >= 0; win = win - 1) {
    if (u32(win) != (num_windows - 1u)) {
      for (var step = 0u; step < window; step = step + 1u) {
        acc = g2_double_jac(acc);
      }
    }
    let point = g2_load_from(0u, i * num_windows + u32(win));
    if (g2_affine_is_infinity(point)) {
      continue;
    }
    acc = g2_add_mixed(acc, point);
  }
  g2_store(i, g2_jac_to_affine(acc));
}
