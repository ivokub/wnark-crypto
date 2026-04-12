struct Fp {
  limbs: array<u32, 12>,
}

struct G1Point {
  x: Fp,
  y: Fp,
  z: Fp,
}

struct Fp24 {
  limbs: array<u32, 24>,
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
const FP_LIMB16_MASK: u32 = 0xffffu;
const FP_QINV_NEG_16: u32 = 0xfffdu;

const FP_MODULUS16: array<u32, 24> = array<u32, 24>(
  0xaaabu, 0xffffu,
  0xffffu, 0xb9feu,
  0xffffu, 0xb153u,
  0xfffeu, 0x1eabu,
  0xf624u, 0xf6b0u,
  0xd2a0u, 0x6730u,
  0x12bfu, 0xf385u,
  0x4b84u, 0x6477u,
  0xacd7u, 0x434bu,
  0xa7b6u, 0x4b1bu,
  0xe69au, 0x397fu,
  0x11eau, 0x1a01u,
);

const FP_MODULUS_MINUS_TWO: array<u32, 12> = array<u32, 12>(
  0xffffaaa9u,
  0xb9feffffu,
  0xb153ffffu,
  0x1eabfffeu,
  0xf6b0f624u,
  0x6730d2a0u,
  0xf38512bfu,
  0x64774b84u,
  0x434bacd7u,
  0x4b1ba7b6u,
  0x397fe69au,
  0x1a0111eau,
);

@group(0) @binding(0) var<storage, read> input_a: array<u32>;
@group(0) @binding(1) var<storage, read> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn fp_zero() -> Fp {
  var z: Fp;
  for (var i = 0u; i < 12u; i = i + 1u) {
    z.limbs[i] = 0u;
  }
  return z;
}

fn fp_one() -> Fp {
  var z: Fp;
  z.limbs[0] = 0x0002fffdu;
  z.limbs[1] = 0x76090000u;
  z.limbs[2] = 0xc40c0002u;
  z.limbs[3] = 0xebf4000bu;
  z.limbs[4] = 0x53c758bau;
  z.limbs[5] = 0x5f489857u;
  z.limbs[6] = 0x70525745u;
  z.limbs[7] = 0x77ce5853u;
  z.limbs[8] = 0xa256ec6du;
  z.limbs[9] = 0x5c071a97u;
  z.limbs[10] = 0xfa80e493u;
  z.limbs[11] = 0x15f65ec3u;
  return z;
}

fn fp_modulus() -> Fp {
  var z: Fp;
  z.limbs[0] = 0xffffaaabu;
  z.limbs[1] = 0xb9feffffu;
  z.limbs[2] = 0xb153ffffu;
  z.limbs[3] = 0x1eabfffeu;
  z.limbs[4] = 0xf6b0f624u;
  z.limbs[5] = 0x6730d2a0u;
  z.limbs[6] = 0xf38512bfu;
  z.limbs[7] = 0x64774b84u;
  z.limbs[8] = 0x434bacd7u;
  z.limbs[9] = 0x4b1ba7b6u;
  z.limbs[10] = 0x397fe69au;
  z.limbs[11] = 0x1a0111eau;
  return z;
}

fn adc(a: u32, b: u32, carry: u32) -> vec2<u32> {
  let sum0 = a + b;
  let carry0 = select(0u, 1u, sum0 < a);
  let sum1 = sum0 + carry;
  let carry1 = select(0u, 1u, sum1 < sum0);
  return vec2<u32>(sum1, carry0 | carry1);
}

fn sbb(a: u32, b: u32, borrow: u32) -> vec2<u32> {
  let diff0 = a - b;
  let borrow0 = select(0u, 1u, a < b);
  let diff1 = diff0 - borrow;
  let borrow1 = select(0u, 1u, diff1 > diff0);
  return vec2<u32>(diff1, borrow0 | borrow1);
}

fn fp_is_zero(x: Fp) -> bool {
  var acc = 0u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let limb = x.limbs[i];
    acc = acc | limb;
  }
  return acc == 0u;
}

fn fp_equal(x: Fp, y: Fp) -> bool {
  for (var i = 0u; i < 12u; i = i + 1u) {
    let x_limb = x.limbs[i];
    let y_limb = y.limbs[i];
    if (x_limb != y_limb) {
      return false;
    }
  }
  return true;
}

fn fp_gte(x: Fp, y: Fp) -> bool {
  for (var i: i32 = 11; i >= 0; i = i - 1) {
    let idx = u32(i);
    let xv = x.limbs[idx];
    let yv = y.limbs[idx];
    if (xv != yv) {
      return xv > yv;
    }
  }
  return true;
}

fn fp_add_modulus(x: Fp) -> Fp {
  let q = fp_modulus();
  var z: Fp;
  var carry = 0u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let lane = adc(x.limbs[i], q.limbs[i], carry);
    z.limbs[i] = lane.x;
    carry = lane.y;
  }
  return z;
}

fn fp_sub_modulus(x: Fp) -> Fp {
  let q = fp_modulus();
  var z: Fp;
  var borrow = 0u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let lane = sbb(x.limbs[i], q.limbs[i], borrow);
    z.limbs[i] = lane.x;
    borrow = lane.y;
  }
  return z;
}

fn fp_add(x: Fp, y: Fp) -> Fp {
  var z: Fp;
  var carry = 0u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let lane = adc(x.limbs[i], y.limbs[i], carry);
    z.limbs[i] = lane.x;
    carry = lane.y;
  }
  if ((carry != 0u) || fp_gte(z, fp_modulus())) {
    return fp_sub_modulus(z);
  }
  return z;
}

fn fp_sub(x: Fp, y: Fp) -> Fp {
  var z: Fp;
  var borrow = 0u;
  for (var i = 0u; i < 12u; i = i + 1u) {
    let lane = sbb(x.limbs[i], y.limbs[i], borrow);
    z.limbs[i] = lane.x;
    borrow = lane.y;
  }
  if (borrow != 0u) {
    return fp_add_modulus(z);
  }
  return z;
}

fn fp_neg(x: Fp) -> Fp {
  if (fp_is_zero(x)) {
    return fp_zero();
  }
  return fp_sub(fp_modulus(), x);
}

fn fp_double(x: Fp) -> Fp {
  return fp_add(x, x);
}

fn fp_unpack16(x: Fp) -> Fp24 {
  var z: Fp24;
  for (var i = 0u; i < 12u; i = i + 1u) {
    z.limbs[2u * i] = x.limbs[i] & FP_LIMB16_MASK;
    z.limbs[2u * i + 1u] = x.limbs[i] >> 16u;
  }
  return z;
}

fn fp_pack16(x: Fp24) -> Fp {
  var z: Fp;
  for (var i = 0u; i < 12u; i = i + 1u) {
    z.limbs[i] = x.limbs[2u * i] | (x.limbs[2u * i + 1u] << 16u);
  }
  return z;
}

fn fp24_gte_modulus(x: Fp24) -> bool {
  for (var i: i32 = 23; i >= 0; i = i - 1) {
    let idx = u32(i);
    let xv = x.limbs[idx];
    let qv = FP_MODULUS16[idx];
    if (xv != qv) {
      return xv > qv;
    }
  }
  return true;
}

fn fp24_sub_modulus(x: Fp24) -> Fp24 {
  var z: Fp24;
  var borrow = 0u;
  for (var i = 0u; i < 24u; i = i + 1u) {
    let lane = sbb(x.limbs[i], FP_MODULUS16[i], borrow);
    z.limbs[i] = lane.x & FP_LIMB16_MASK;
    borrow = lane.y;
  }
  return z;
}

fn fp_mul(x: Fp, y: Fp) -> Fp {
  let a = fp_unpack16(x);
  let b = fp_unpack16(y);
  var t: array<u32, 25>;

  for (var i = 0u; i < 24u; i = i + 1u) {
    var carry = 0u;
    let bi = b.limbs[i];
    for (var j = 0u; j < 24u; j = j + 1u) {
      let a_limb = a.limbs[j];
      let uv = t[j] + (a_limb * bi) + carry;
      t[j] = uv & FP_LIMB16_MASK;
      carry = uv >> 16u;
    }
    t[24] = carry;

    let m = (t[0] * FP_QINV_NEG_16) & FP_LIMB16_MASK;
    carry = 0u;
    for (var j = 0u; j < 24u; j = j + 1u) {
      let q_limb = FP_MODULUS16[j];
      let uv = t[j] + (m * q_limb) + carry;
      if (j > 0u) {
        t[j - 1u] = uv & FP_LIMB16_MASK;
      }
      carry = uv >> 16u;
    }
    let uv = t[24] + carry;
    t[23] = uv & FP_LIMB16_MASK;
    t[24] = uv >> 16u;
  }

  var z24: Fp24;
  for (var i = 0u; i < 24u; i = i + 1u) {
    z24.limbs[i] = t[i];
  }
  if ((t[24] != 0u) || fp24_gte_modulus(z24)) {
    z24 = fp24_sub_modulus(z24);
  }
  return fp_pack16(z24);
}

fn fp_square(x: Fp) -> Fp {
  return fp_mul(x, x);
}

fn fp_inverse(x: Fp) -> Fp {
  if (fp_is_zero(x)) {
    return fp_zero();
  }
  var acc = fp_one();
  for (var word_index: i32 = 11; word_index >= 0; word_index = word_index - 1) {
    let word = FP_MODULUS_MINUS_TWO[u32(word_index)];
    for (var bit_index: i32 = 31; bit_index >= 0; bit_index = bit_index - 1) {
      acc = fp_square(acc);
      if (((word >> u32(bit_index)) & 1u) != 0u) {
        acc = fp_mul(acc, x);
      }
    }
  }
  return acc;
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
