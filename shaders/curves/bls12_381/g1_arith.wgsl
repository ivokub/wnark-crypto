struct G1Point {
  x: Fp,
  y: Fp,
  z: Fp,
}

struct Params {
  count: u32,
  opcode: u32,
  _pad0: u32,
  _pad1: u32,
}

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
  if (i >= params.count) {
    return;
  }
  g1_store(i, g1_dispatch(params.opcode, g1_load_from(0u, i), g1_load_from(1u, i)));
}
