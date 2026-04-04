struct Fp {
  limbs: array<u32, 8>,
}

struct G1Point {
  x: Fp,
  y: Fp,
  z: Fp,
}

struct Fp16 {
  limbs: array<u32, 16>,
}

struct Params {
  lane0: vec4<u32>,
  lane1: vec4<u32>,
}

var<workgroup> g1_window_shared: array<G1Point, 64>;

const FP_OP_COPY: u32 = 0u;
const FP_OP_ZERO: u32 = 1u;
const FP_OP_ONE: u32 = 2u;
const FP_OP_ADD: u32 = 3u;
const FP_OP_SUB: u32 = 4u;
const FP_OP_NEG: u32 = 5u;
const FP_OP_DOUBLE: u32 = 6u;
const FP_OP_NORMALIZE: u32 = 7u;
const FP_OP_EQUAL: u32 = 8u;
const FP_OP_MUL: u32 = 9u;
const FP_OP_SQUARE: u32 = 10u;
const FP_OP_TO_MONT: u32 = 11u;
const FP_OP_FROM_MONT: u32 = 12u;
const G1_OP_COPY: u32 = 0u;
const G1_OP_JAC_INFINITY: u32 = 1u;
const G1_OP_AFFINE_TO_JAC: u32 = 2u;
const G1_OP_NEG_JAC: u32 = 3u;
const G1_OP_DOUBLE_JAC: u32 = 4u;
const G1_OP_ADD_MIXED: u32 = 5u;
const G1_OP_JAC_TO_AFFINE: u32 = 6u;
const G1_OP_AFFINE_ADD: u32 = 7u;
const G1_OP_SCALAR_MUL_AFFINE: u32 = 8u;
const FP_LIMB16_MASK: u32 = 0xffffu;
const FP_QINV_NEG_16: u32 = 0x6389u;

const FP_MODULUS16: array<u32, 16> = array<u32, 16>(
  0xfd47u, 0xd87cu,
  0x8c16u, 0x3c20u,
  0xca8du, 0x6871u,
  0x6a91u, 0x9781u,
  0x585du, 0x8181u,
  0x45b6u, 0xb850u,
  0xa029u, 0xe131u,
  0x4e72u, 0x3064u,
);

const FP_MODULUS_MINUS_TWO: array<u32, 8> = array<u32, 8>(
  0xd87cfd45u,
  0x3c208c16u,
  0x6871ca8du,
  0x97816a91u,
  0x8181585du,
  0xb85045b6u,
  0xe131a029u,
  0x30644e72u,
);

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

fn fp_zero() -> Fp {
  var z: Fp;
  z.limbs[0] = 0u;
  z.limbs[1] = 0u;
  z.limbs[2] = 0u;
  z.limbs[3] = 0u;
  z.limbs[4] = 0u;
  z.limbs[5] = 0u;
  z.limbs[6] = 0u;
  z.limbs[7] = 0u;
  return z;
}

fn fp_one() -> Fp {
  var z: Fp;
  z.limbs[0] = 0xc58f0d9du;
  z.limbs[1] = 0xd35d438du;
  z.limbs[2] = 0xf5c70b3du;
  z.limbs[3] = 0x0a78eb28u;
  z.limbs[4] = 0x7879462cu;
  z.limbs[5] = 0x666ea36fu;
  z.limbs[6] = 0x9a07df2fu;
  z.limbs[7] = 0x0e0a77c1u;
  return z;
}

fn fp_one_regular() -> Fp {
  var z = fp_zero();
  z.limbs[0] = 1u;
  return z;
}

fn fp_rsquare_regular() -> Fp {
  var z: Fp;
  z.limbs[0] = 0x538afa89u;
  z.limbs[1] = 0xf32cfc5bu;
  z.limbs[2] = 0xd44501fbu;
  z.limbs[3] = 0xb5e71911u;
  z.limbs[4] = 0x0a417ff6u;
  z.limbs[5] = 0x47ab1effu;
  z.limbs[6] = 0xcab8351fu;
  z.limbs[7] = 0x06d89f71u;
  return z;
}

fn fp_modulus() -> Fp {
  var z: Fp;
  z.limbs[0] = 0xd87cfd47u;
  z.limbs[1] = 0x3c208c16u;
  z.limbs[2] = 0x6871ca8du;
  z.limbs[3] = 0x97816a91u;
  z.limbs[4] = 0x8181585du;
  z.limbs[5] = 0xb85045b6u;
  z.limbs[6] = 0xe131a029u;
  z.limbs[7] = 0x30644e72u;
  return z;
}

fn fp_predicate(value: bool) -> Fp {
  var z = fp_zero();
  if (value) {
    z = fp_one();
  }
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
  return (x.limbs[0] | x.limbs[1] | x.limbs[2] | x.limbs[3] |
    x.limbs[4] | x.limbs[5] | x.limbs[6] | x.limbs[7]) == 0u;
}

fn fp_equal(x: Fp, y: Fp) -> bool {
  return (x.limbs[0] == y.limbs[0]) &&
    (x.limbs[1] == y.limbs[1]) &&
    (x.limbs[2] == y.limbs[2]) &&
    (x.limbs[3] == y.limbs[3]) &&
    (x.limbs[4] == y.limbs[4]) &&
    (x.limbs[5] == y.limbs[5]) &&
    (x.limbs[6] == y.limbs[6]) &&
    (x.limbs[7] == y.limbs[7]);
}

fn fp_gte(x: Fp, y: Fp) -> bool {
  if (x.limbs[7] != y.limbs[7]) {
    return x.limbs[7] > y.limbs[7];
  }
  if (x.limbs[6] != y.limbs[6]) {
    return x.limbs[6] > y.limbs[6];
  }
  if (x.limbs[5] != y.limbs[5]) {
    return x.limbs[5] > y.limbs[5];
  }
  if (x.limbs[4] != y.limbs[4]) {
    return x.limbs[4] > y.limbs[4];
  }
  if (x.limbs[3] != y.limbs[3]) {
    return x.limbs[3] > y.limbs[3];
  }
  if (x.limbs[2] != y.limbs[2]) {
    return x.limbs[2] > y.limbs[2];
  }
  if (x.limbs[1] != y.limbs[1]) {
    return x.limbs[1] > y.limbs[1];
  }
  return x.limbs[0] >= y.limbs[0];
}

fn fp_add_modulus(x: Fp) -> Fp {
  let q = fp_modulus();
  var z: Fp;
  var carry = 0u;
  var lane = adc(x.limbs[0], q.limbs[0], carry);
  z.limbs[0] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[1], q.limbs[1], carry);
  z.limbs[1] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[2], q.limbs[2], carry);
  z.limbs[2] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[3], q.limbs[3], carry);
  z.limbs[3] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[4], q.limbs[4], carry);
  z.limbs[4] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[5], q.limbs[5], carry);
  z.limbs[5] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[6], q.limbs[6], carry);
  z.limbs[6] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[7], q.limbs[7], carry);
  z.limbs[7] = lane.x;
  return z;
}

fn fp_sub_modulus(x: Fp) -> Fp {
  let q = fp_modulus();
  var z: Fp;
  var borrow = 0u;
  var lane = sbb(x.limbs[0], q.limbs[0], borrow);
  z.limbs[0] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[1], q.limbs[1], borrow);
  z.limbs[1] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[2], q.limbs[2], borrow);
  z.limbs[2] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[3], q.limbs[3], borrow);
  z.limbs[3] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[4], q.limbs[4], borrow);
  z.limbs[4] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[5], q.limbs[5], borrow);
  z.limbs[5] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[6], q.limbs[6], borrow);
  z.limbs[6] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[7], q.limbs[7], borrow);
  z.limbs[7] = lane.x;
  return z;
}

fn fp_add(x: Fp, y: Fp) -> Fp {
  var z: Fp;
  var carry = 0u;
  var lane = adc(x.limbs[0], y.limbs[0], carry);
  z.limbs[0] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[1], y.limbs[1], carry);
  z.limbs[1] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[2], y.limbs[2], carry);
  z.limbs[2] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[3], y.limbs[3], carry);
  z.limbs[3] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[4], y.limbs[4], carry);
  z.limbs[4] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[5], y.limbs[5], carry);
  z.limbs[5] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[6], y.limbs[6], carry);
  z.limbs[6] = lane.x;
  carry = lane.y;
  lane = adc(x.limbs[7], y.limbs[7], carry);
  z.limbs[7] = lane.x;
  if ((lane.y != 0u) || fp_gte(z, fp_modulus())) {
    return fp_sub_modulus(z);
  }
  return z;
}

fn fp_sub(x: Fp, y: Fp) -> Fp {
  var z: Fp;
  var borrow = 0u;
  var lane = sbb(x.limbs[0], y.limbs[0], borrow);
  z.limbs[0] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[1], y.limbs[1], borrow);
  z.limbs[1] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[2], y.limbs[2], borrow);
  z.limbs[2] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[3], y.limbs[3], borrow);
  z.limbs[3] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[4], y.limbs[4], borrow);
  z.limbs[4] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[5], y.limbs[5], borrow);
  z.limbs[5] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[6], y.limbs[6], borrow);
  z.limbs[6] = lane.x;
  borrow = lane.y;
  lane = sbb(x.limbs[7], y.limbs[7], borrow);
  z.limbs[7] = lane.x;
  if (lane.y != 0u) {
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

fn fp_normalize(x: Fp) -> Fp {
  if (fp_gte(x, fp_modulus())) {
    return fp_sub_modulus(x);
  }
  return x;
}

fn fp_unpack16(x: Fp) -> Fp16 {
  var z: Fp16;
  for (var i = 0u; i < 8u; i = i + 1u) {
    z.limbs[2u * i] = x.limbs[i] & FP_LIMB16_MASK;
    z.limbs[2u * i + 1u] = x.limbs[i] >> 16u;
  }
  return z;
}

fn fp_pack16(x: Fp16) -> Fp {
  var z: Fp;
  for (var i = 0u; i < 8u; i = i + 1u) {
    z.limbs[i] = x.limbs[2u * i] | (x.limbs[2u * i + 1u] << 16u);
  }
  return z;
}

fn fp16_gte_modulus(x: Fp16) -> bool {
  for (var i: i32 = 15; i >= 0; i = i - 1) {
    let idx = u32(i);
    let xLimb = x.limbs[idx];
    let qLimb = FP_MODULUS16[idx];
    if (xLimb != qLimb) {
      return xLimb > qLimb;
    }
  }
  return true;
}

fn fp16_sub_modulus(x: Fp16) -> Fp16 {
  var z: Fp16;
  var borrow = 0u;
  for (var i = 0u; i < 16u; i = i + 1u) {
    let lane = sbb(x.limbs[i], FP_MODULUS16[i], borrow);
    z.limbs[i] = lane.x & FP_LIMB16_MASK;
    borrow = lane.y;
  }
  return z;
}

fn fp_mul(x: Fp, y: Fp) -> Fp {
  let a = fp_unpack16(x);
  let b = fp_unpack16(y);
  var t: array<u32, 17>;

  for (var i = 0u; i < 16u; i = i + 1u) {
    var carry = 0u;
    let bi = b.limbs[i];
    for (var j = 0u; j < 16u; j = j + 1u) {
      let aLimb = a.limbs[j];
      let uv = t[j] + (aLimb * bi) + carry;
      t[j] = uv & FP_LIMB16_MASK;
      carry = uv >> 16u;
    }
    t[16] = carry;

    let m = (t[0] * FP_QINV_NEG_16) & FP_LIMB16_MASK;
    carry = 0u;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let qLimb = FP_MODULUS16[j];
      let uv = t[j] + (m * qLimb) + carry;
      if (j > 0u) {
        t[j - 1u] = uv & FP_LIMB16_MASK;
      }
      carry = uv >> 16u;
    }
    let uv = t[16] + carry;
    t[15] = uv & FP_LIMB16_MASK;
    t[16] = uv >> 16u;
  }

  var z16: Fp16;
  for (var i = 0u; i < 16u; i = i + 1u) {
    z16.limbs[i] = t[i];
  }
  if ((t[16] != 0u) || fp16_gte_modulus(z16)) {
    z16 = fp16_sub_modulus(z16);
  }
  return fp_pack16(z16);
}

fn fp_square(x: Fp) -> Fp {
  return fp_mul(x, x);
}

fn fp_inverse(x: Fp) -> Fp {
  if (fp_is_zero(x)) {
    return fp_zero();
  }
  var acc = fp_one();
  for (var wordIndex: i32 = 7; wordIndex >= 0; wordIndex = wordIndex - 1) {
    let word = FP_MODULUS_MINUS_TWO[u32(wordIndex)];
    for (var bitIndex: i32 = 31; bitIndex >= 0; bitIndex = bitIndex - 1) {
      acc = fp_square(acc);
      if (((word >> u32(bitIndex)) & 1u) != 0u) {
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

fn g1_neg_affine(q: G1Point) -> G1Point {
  if (g1_affine_is_infinity(q)) {
    return q;
  }
  var p = q;
  p.y = fp_neg(q.y);
  return p;
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

fn g1_scalar_mul_affine_jac(base: G1Point, scalar_words: Fp) -> G1Point {
  var acc = g1_jac_infinity();
  for (var bit: i32 = 255; bit >= 0; bit = bit - 1) {
    acc = g1_double_jac(acc);
    if (g1_scalar_bit(scalar_words, u32(bit))) {
      acc = g1_add_mixed(acc, base);
    }
  }
  return acc;
}

fn g1_scalar_mul_affine(base: G1Point, scalar_words: Fp) -> G1Point {
  if (g1_affine_is_infinity(base)) {
    return g1_jac_to_affine(g1_jac_infinity());
  }
  let acc = g1_scalar_mul_affine_jac(base, scalar_words);
  return g1_jac_to_affine(acc);
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
  if (opcode == G1_OP_SCALAR_MUL_AFFINE) {
    return g1_scalar_mul_affine(a, b.x);
  }
  return g1_jac_infinity();
}

fn fp_load_from(buffer_kind: u32, base: u32) -> Fp {
  var z: Fp;
  if (buffer_kind == 0u) {
    z.limbs[0] = input_a[base + 0u];
    z.limbs[1] = input_a[base + 1u];
    z.limbs[2] = input_a[base + 2u];
    z.limbs[3] = input_a[base + 3u];
    z.limbs[4] = input_a[base + 4u];
    z.limbs[5] = input_a[base + 5u];
    z.limbs[6] = input_a[base + 6u];
    z.limbs[7] = input_a[base + 7u];
    return z;
  }
  z.limbs[0] = input_b[base + 0u];
  z.limbs[1] = input_b[base + 1u];
  z.limbs[2] = input_b[base + 2u];
  z.limbs[3] = input_b[base + 3u];
  z.limbs[4] = input_b[base + 4u];
  z.limbs[5] = input_b[base + 5u];
  z.limbs[6] = input_b[base + 6u];
  z.limbs[7] = input_b[base + 7u];
  return z;
}

fn g1_load_from(buffer_kind: u32, index: u32) -> G1Point {
  let base = index * 24u;
  var p: G1Point;
  p.x = fp_load_from(buffer_kind, base + 0u);
  p.y = fp_load_from(buffer_kind, base + 8u);
  p.z = fp_load_from(buffer_kind, base + 16u);
  return p;
}

fn fp_store(base: u32, value: Fp) {
  output[base + 0u] = value.limbs[0];
  output[base + 1u] = value.limbs[1];
  output[base + 2u] = value.limbs[2];
  output[base + 3u] = value.limbs[3];
  output[base + 4u] = value.limbs[4];
  output[base + 5u] = value.limbs[5];
  output[base + 6u] = value.limbs[6];
  output[base + 7u] = value.limbs[7];
}

fn g1_store(index: u32, value: G1Point) {
  let base = index * 24u;
  fp_store(base + 0u, value.x);
  fp_store(base + 8u, value.y);
  fp_store(base + 16u, value.z);
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
fn g1_msm_bucket_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let bucket_count = params_bucket_count();
  let num_windows = params_num_windows();
  let per_instance = num_windows * bucket_count;
  let instance = i / per_instance;
  let rem = i % per_instance;
  let win = rem / bucket_count;
  let bucket = (rem % bucket_count) + 1u;
  let window = params_window();
  let terms_per_instance = params_terms_per_instance();
  let bit_offset = win * window;
  let base_offset = instance * terms_per_instance;

  var acc = g1_jac_infinity();
  for (var term = 0u; term < terms_per_instance; term = term + 1u) {
    let idx = base_offset + term;
    let scalar_words = g1_load_from(1u, idx).x;
    if (g1_window_digit(scalar_words, bit_offset, window) == bucket) {
      acc = g1_add_mixed(acc, g1_load_from(0u, idx));
    }
  }
  g1_store(i, g1_jac_to_affine(acc));
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
fn g1_msm_window_weight_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let bucket_count = params_bucket_count();
  let bucket = (i % bucket_count) + 1u;
  let point = g1_load_from(0u, i);
  if (g1_affine_is_infinity(point)) {
    g1_store(i, point);
    return;
  }
  g1_store(i, g1_scalar_mul_affine_small(point, bucket));
}

@compute @workgroup_size(64)
fn g1_msm_window_reduce_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params_count()) {
    return;
  }
  let row_width = params_row_width();
  let next_width = (row_width + 1u) / 2u;
  let row = i / next_width;
  let slot = i % next_width;
  let left_index = row * row_width + (2u * slot);
  let right_index = left_index + 1u;
  let left = g1_load_from(0u, left_index);
  var right: G1Point;
  if ((2u * slot + 1u) < row_width) {
    right = g1_load_from(0u, right_index);
  } else {
    right.x = fp_zero();
    right.y = fp_zero();
    right.z = fp_zero();
  }
  g1_store(i, g1_jac_to_affine(g1_add_mixed(g1_affine_to_jac(left), right)));
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
