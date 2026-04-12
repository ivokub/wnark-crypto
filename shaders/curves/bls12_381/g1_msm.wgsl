struct G1Point {
  x: Fp,
  y: Fp,
  z: Fp,
}

struct Params {
  lane0: vec4<u32>,
  lane1: vec4<u32>,
}

var<workgroup> g1_window_shared: array<G1Point, 64>;

const G1_OP_COPY: u32 = 0u;
const G1_OP_JAC_INFINITY: u32 = 1u;
const G1_OP_AFFINE_TO_JAC: u32 = 2u;
const G1_OP_NEG_JAC: u32 = 3u;
const G1_OP_DOUBLE_JAC: u32 = 4u;
const G1_OP_ADD_MIXED: u32 = 5u;
const G1_OP_JAC_TO_AFFINE: u32 = 6u;
const G1_OP_AFFINE_ADD: u32 = 7u;

@group(0) @binding(0) var<storage, read> input_a: array<u32>;
@group(0) @binding(1) var<storage, read> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> input_meta0: array<u32>;
@group(0) @binding(5) var<storage, read> input_meta1: array<u32>;
@group(0) @binding(6) var<storage, read> input_meta2: array<u32>;

fn params_count() -> u32 {
  return params.lane0.x;
}

fn params_opcode() -> u32 {
  return params.lane0.y;
}

fn params_terms_per_instance() -> u32 {
  return params.lane0.z;
}

fn params_window() -> u32 {
  return params.lane0.w;
}

fn params_num_windows() -> u32 {
  return params.lane1.x;
}

fn params_bucket_count() -> u32 {
  return params.lane1.y;
}

fn params_row_width() -> u32 {
  return params.lane1.z;
}

fn g1_jac_infinity() -> G1Point {
  var p: G1Point;
  p.x = fp_one();
  p.y = fp_one();
  p.z = fp_zero();
  return p;
}

fn g1_affine_is_infinity(a: G1Point) -> bool {
  return fp_is_zero(a.z);
}

fn g1_jac_is_infinity(p: G1Point) -> bool {
  return fp_is_zero(p.z);
}

fn g1_affine_to_jac(a: G1Point) -> G1Point {
  if (g1_affine_is_infinity(a)) {
    return g1_jac_infinity();
  }
  var p: G1Point;
  p.x = a.x;
  p.y = a.y;
  p.z = fp_one();
  return p;
}

fn g1_jac_to_affine(p: G1Point) -> G1Point {
  if (g1_jac_is_infinity(p)) {
    var inf: G1Point;
    inf.x = fp_zero();
    inf.y = fp_zero();
    inf.z = fp_zero();
    return inf;
  }
  let a = fp_inverse(p.z);
  let b = fp_square(a);
  var out: G1Point;
  out.x = fp_mul(p.x, b);
  out.y = fp_mul(fp_mul(p.y, b), a);
  out.z = fp_one();
  return out;
}

fn g1_neg_jac(q: G1Point) -> G1Point {
  var p = q;
  p.y = fp_neg(q.y);
  return p;
}

fn g1_neg_affine(q: G1Point) -> G1Point {
  var p = q;
  p.y = fp_neg(q.y);
  return p;
}

fn g1_double_mixed(a: G1Point) -> G1Point {
  if (g1_affine_is_infinity(a)) {
    return g1_jac_infinity();
  }
  var xx = fp_square(a.x);
  let yy = fp_square(a.y);
  var yyyy = fp_square(yy);
  var s = fp_add(a.x, yy);
  s = fp_square(s);
  s = fp_sub(s, xx);
  s = fp_sub(s, yyyy);
  s = fp_double(s);
  var m = fp_double(xx);
  m = fp_add(m, xx);
  let t = fp_sub(fp_sub(fp_square(m), s), s);

  var p: G1Point;
  p.x = t;
  p.y = fp_mul(fp_sub(s, t), m);
  yyyy = fp_double(fp_double(fp_double(yyyy)));
  p.y = fp_sub(p.y, yyyy);
  p.z = fp_double(a.y);
  return p;
}

fn g1_double_jac(q: G1Point) -> G1Point {
  var a = fp_square(q.x);
  let b = fp_square(q.y);
  let c = fp_square(b);
  var d = fp_add(q.x, b);
  d = fp_square(d);
  d = fp_sub(d, a);
  d = fp_sub(d, c);
  d = fp_double(d);
  var e = fp_double(a);
  e = fp_add(e, a);
  let f = fp_square(e);
  var t = fp_double(d);

  var p: G1Point;
  p.z = fp_double(fp_mul(q.y, q.z));
  p.x = fp_sub(f, t);
  p.y = fp_mul(fp_sub(d, p.x), e);
  t = fp_double(fp_double(fp_double(c)));
  p.y = fp_sub(p.y, t);
  return p;
}

fn g1_add_mixed(p: G1Point, a: G1Point) -> G1Point {
  if (g1_affine_is_infinity(a)) {
    return p;
  }
  if (g1_jac_is_infinity(p)) {
    return g1_affine_to_jac(a);
  }

  let z1z1 = fp_square(p.z);
  let u2 = fp_mul(a.x, z1z1);
  let s2 = fp_mul(fp_mul(a.y, p.z), z1z1);

  if (fp_equal(u2, p.x) && fp_equal(s2, p.y)) {
    return g1_double_mixed(a);
  }

  let h = fp_sub(u2, p.x);
  let hh = fp_square(h);
  let i = fp_double(fp_double(hh));
  let j = fp_mul(h, i);
  let r = fp_double(fp_sub(s2, p.y));
  let v = fp_mul(p.x, i);

  var out: G1Point;
  out.x = fp_sub(fp_sub(fp_sub(fp_square(r), j), v), v);
  out.y = fp_sub(fp_mul(fp_sub(v, out.x), r), fp_double(fp_mul(j, p.y)));
  out.z = fp_square(fp_add(p.z, h));
  out.z = fp_sub(out.z, z1z1);
  out.z = fp_sub(out.z, hh);
  return out;
}

fn g1_add_affine(a: G1Point, b: G1Point) -> G1Point {
  return g1_jac_to_affine(g1_add_mixed(g1_affine_to_jac(a), b));
}

fn g1_scalar_bit(words: Fp, bit: u32) -> bool {
  let word = words.limbs[bit / 32u];
  return ((word >> (bit % 32u)) & 1u) != 0u;
}

fn g1_window_digit(words: Fp, bit_offset: u32, window: u32) -> u32 {
  if (window == 0u) {
    return 0u;
  }
  let word = bit_offset / 32u;
  let shift = bit_offset % 32u;
  let mask = (1u << window) - 1u;
  if (word >= 8u) {
    return 0u;
  }
  if ((shift + window) <= 32u) {
    return (words.limbs[word] >> shift) & mask;
  }
  let low_bits = words.limbs[word] >> shift;
  if ((word + 1u) >= 8u) {
    return low_bits & mask;
  }
  let high_width = shift + window - 32u;
  let high_mask = (1u << high_width) - 1u;
  let high_bits = words.limbs[word + 1u] & high_mask;
  return (low_bits | (high_bits << (32u - shift))) & mask;
}

fn g1_scalar_mul_affine_small(base: G1Point, scalar: u32) -> G1Point {
  if (scalar == 0u || g1_affine_is_infinity(base)) {
    return g1_jac_to_affine(g1_jac_infinity());
  }
  var acc = g1_jac_infinity();
  var cur_jac = g1_affine_to_jac(base);
  var cur_aff = base;
  var k = scalar;
  loop {
    if ((k & 1u) != 0u) {
      acc = g1_add_mixed(acc, cur_aff);
    }
    k = k >> 1u;
    if (k == 0u) {
      break;
    }
    cur_jac = g1_double_jac(cur_jac);
    cur_aff = g1_jac_to_affine(cur_jac);
  }
  return g1_jac_to_affine(acc);
}

fn g1_dispatch(opcode: u32, a: G1Point, b: G1Point) -> G1Point {
  if (opcode == G1_OP_COPY) {
    return a;
  }
  if (opcode == G1_OP_JAC_INFINITY) {
    return g1_jac_infinity();
  }
  if (opcode == G1_OP_AFFINE_TO_JAC) {
    return g1_affine_to_jac(a);
  }
  if (opcode == G1_OP_NEG_JAC) {
    return g1_neg_jac(a);
  }
  if (opcode == G1_OP_DOUBLE_JAC) {
    return g1_double_jac(a);
  }
  if (opcode == G1_OP_ADD_MIXED) {
    return g1_add_mixed(a, b);
  }
  if (opcode == G1_OP_JAC_TO_AFFINE) {
    return g1_jac_to_affine(a);
  }
  if (opcode == G1_OP_AFFINE_ADD) {
    return g1_jac_to_affine(g1_add_mixed(g1_affine_to_jac(a), b));
  }
  return g1_jac_infinity();
}

fn fp_load_from(buffer_kind: u32, base: u32) -> Fp {
  var z: Fp;
  if (buffer_kind == 0u) {
    for (var i = 0u; i < 12u; i = i + 1u) {
      z.limbs[i] = input_a[base + i];
    }
    return z;
  }
  for (var i = 0u; i < 12u; i = i + 1u) {
    z.limbs[i] = input_b[base + i];
  }
  return z;
}

fn g1_load_from(buffer_kind: u32, index: u32) -> G1Point {
  let base = index * 36u;
  var p: G1Point;
  p.x = fp_load_from(buffer_kind, base + 0u);
  p.y = fp_load_from(buffer_kind, base + 12u);
  p.z = fp_load_from(buffer_kind, base + 24u);
  return p;
}

fn fp_store(base: u32, value: Fp) {
  for (var i = 0u; i < 12u; i = i + 1u) {
    output[base + i] = value.limbs[i];
  }
}

fn g1_store(index: u32, value: G1Point) {
  let base = index * 36u;
  fp_store(base + 0u, value.x);
  fp_store(base + 12u, value.y);
  fp_store(base + 24u, value.z);
}

@compute @workgroup_size(64)
fn g1_ops_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  g1_store(i, g1_dispatch(params_opcode(), g1_load_from(0u, i), g1_load_from(1u, i)));
}

@compute @workgroup_size(64)
fn g1_msm_bucket_sparse_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let start = input_meta1[i];
  let size = input_meta2[i];
  var acc = g1_jac_infinity();
  for (var j = 0u; j < size; j = j + 1u) {
    let raw = input_meta0[start + j];
    let idx = raw & 0x7fffffffu;
    let neg = (raw & 0x80000000u) != 0u;
    var point = g1_load_from(0u, idx);
    if (neg) {
      point = g1_neg_affine(point);
    }
    acc = g1_add_mixed(acc, point);
  }
  g1_store(i, g1_jac_to_affine(acc));
}

@compute @workgroup_size(64)
fn g1_msm_bucket_merge_main(
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
  var local_sum = g1_jac_infinity();
  var j = tid;
  loop {
    if (j >= count) {
      break;
    }
    let point = g1_load_from(0u, start + j);
    if (!g1_affine_is_infinity(point)) {
      local_sum = g1_add_mixed(local_sum, point);
    }
    j = j + 64u;
  }
  g1_window_shared[tid] = g1_jac_to_affine(local_sum);
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      g1_window_shared[tid] = g1_add_affine(g1_window_shared[tid], g1_window_shared[tid + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    g1_store(i, g1_window_shared[0u]);
  }
}

@compute @workgroup_size(64)
fn g1_msm_window_sparse_main(
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
  var local_sum = g1_jac_infinity();
  var j = tid;
  loop {
    if (j >= count) {
      break;
    }
    let slot = start + j;
    let point = g1_load_from(0u, slot);
    let value = input_meta0[slot];
    if (!g1_affine_is_infinity(point) && value != 0u) {
      let weighted = g1_scalar_mul_affine_small(point, value);
      local_sum = g1_add_mixed(local_sum, weighted);
    }
    j = j + 64u;
  }
  g1_window_shared[tid] = g1_jac_to_affine(local_sum);
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      g1_window_shared[tid] = g1_add_affine(g1_window_shared[tid], g1_window_shared[tid + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    g1_store(i, g1_window_shared[0u]);
  }
}

@compute @workgroup_size(64)
fn g1_msm_weight_buckets_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let point = g1_load_from(0u, i);
  let value = input_meta0[i];
  if (g1_affine_is_infinity(point) || value == 0u) {
    g1_store(i, g1_jac_to_affine(g1_jac_infinity()));
    return;
  }
  g1_store(i, g1_scalar_mul_affine_small(point, value));
}

@compute @workgroup_size(64)
fn g1_msm_subsum_phase1_main(
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
  var local_sum = g1_jac_infinity();
  var j = tid;
  loop {
    if (j >= count) {
      break;
    }
    let point = g1_load_from(0u, start + j);
    if (!g1_affine_is_infinity(point)) {
      local_sum = g1_add_mixed(local_sum, point);
    }
    j = j + 64u;
  }
  g1_window_shared[tid] = g1_jac_to_affine(local_sum);
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      g1_window_shared[tid] = g1_add_affine(g1_window_shared[tid], g1_window_shared[tid + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    g1_store(i, g1_window_shared[0u]);
  }
}

@compute @workgroup_size(64)
fn g1_msm_window_dense_main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
) {
  let i = wg_id.x;
  let tid = local_id.x;
  if (i >= params_count()) {
    return;
  }
  let row_width = params_row_width();
  let bucket_count = params_bucket_count();
  let active_threads = min(row_width, 64u);
  var local_sum = g1_jac_infinity();
  var j = tid;
  loop {
    if (j >= active_threads) {
      break;
    }
    let segment_start = (row_width * j) / active_threads;
    let segment_end = (row_width * (j + 1u)) / active_threads;
    var running = g1_jac_infinity();
    var total = g1_jac_infinity();
    for (var pos = segment_start; pos < segment_end; pos = pos + 1u) {
      let point = g1_load_from(0u, i * bucket_count + pos);
      if (!g1_affine_is_infinity(point)) {
        running = g1_add_mixed(running, point);
      }
      if (!g1_jac_is_infinity(running)) {
        total = g1_add_mixed(total, g1_jac_to_affine(running));
      }
    }
    local_sum = total;
    j = j + 64u;
  }
  g1_window_shared[tid] = g1_jac_to_affine(local_sum);
  workgroupBarrier();

  var stride = 32u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride && (tid + stride) < active_threads) {
      g1_window_shared[tid] = g1_add_affine(g1_window_shared[tid], g1_window_shared[tid + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    g1_store(i, g1_window_shared[0u]);
  }
}

@compute @workgroup_size(64)
fn g1_msm_combine_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let num_windows = params_num_windows();
  let window = params_window();
  var acc = g1_jac_infinity();
  for (var win: i32 = i32(num_windows) - 1; win >= 0; win = win - 1) {
    if (u32(win) != (num_windows - 1u)) {
      for (var step = 0u; step < window; step = step + 1u) {
        acc = g1_double_jac(acc);
      }
    }
    let point = g1_load_from(0u, i * num_windows + u32(win));
    if (g1_affine_is_infinity(point)) {
      continue;
    }
    acc = g1_add_mixed(acc, point);
  }
  g1_store(i, g1_jac_to_affine(acc));
}
