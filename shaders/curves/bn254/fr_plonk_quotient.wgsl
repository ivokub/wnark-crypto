struct PlonkQuotientParams {
  count: u32,
  blind_count: u32,
  _pad0: u32,
  _pad1: u32,
}

const PLONK_FR_WORDS: u32 = 8u;

const PLONK_VEC_L: u32 = 0u;
const PLONK_VEC_R: u32 = 1u;
const PLONK_VEC_O: u32 = 2u;
const PLONK_VEC_Z: u32 = 3u;
const PLONK_VEC_QK: u32 = 4u;
const PLONK_VEC_QL: u32 = 5u;
const PLONK_VEC_QR: u32 = 6u;
const PLONK_VEC_QM: u32 = 7u;
const PLONK_VEC_QO: u32 = 8u;
const PLONK_VEC_S1: u32 = 9u;
const PLONK_VEC_S2: u32 = 10u;
const PLONK_VEC_S3: u32 = 11u;
const PLONK_VEC_TWIDDLES: u32 = 12u;
const PLONK_VEC_DENOMINATORS: u32 = 13u;

const PLONK_BLIND_L: u32 = 0u;
const PLONK_BLIND_R: u32 = 1u;
const PLONK_BLIND_O: u32 = 2u;
const PLONK_BLIND_Z: u32 = 3u;

const PLONK_SCALAR_COSET: u32 = 0u;
const PLONK_SCALAR_LAGRANGE_SCALE: u32 = 1u;
const PLONK_SCALAR_CS: u32 = 2u;
const PLONK_SCALAR_CSS: u32 = 3u;
const PLONK_SCALAR_BETA: u32 = 4u;
const PLONK_SCALAR_GAMMA: u32 = 5u;
const PLONK_SCALAR_ALPHA: u32 = 6u;

@group(0) @binding(0) var<storage, read> plonk_vectors: array<u32>;
@group(0) @binding(1) var<storage, read> plonk_blinds: array<u32>;
@group(0) @binding(2) var<storage, read> plonk_scalars: array<u32>;
@group(0) @binding(3) var<storage, read_write> plonk_output: array<u32>;
@group(0) @binding(4) var<uniform> plonk_params: PlonkQuotientParams;

fn fr_to_mont(x: Fr) -> Fr {
  return fr_mul(x, fr_rsquare_regular());
}

fn fr_from_mont(x: Fr) -> Fr {
  return fr_mul(x, fr_one_regular());
}

fn plonk_load_words(base: u32) -> Fr {
  var z: Fr;
  z.limbs[0] = plonk_vectors[base + 0u];
  z.limbs[1] = plonk_vectors[base + 1u];
  z.limbs[2] = plonk_vectors[base + 2u];
  z.limbs[3] = plonk_vectors[base + 3u];
  z.limbs[4] = plonk_vectors[base + 4u];
  z.limbs[5] = plonk_vectors[base + 5u];
  z.limbs[6] = plonk_vectors[base + 6u];
  z.limbs[7] = plonk_vectors[base + 7u];
  return z;
}

fn plonk_load_vector_mont(vector: u32, index: u32) -> Fr {
  let base = ((vector * plonk_params.count) + index) * PLONK_FR_WORDS;
  return plonk_load_words(base);
}

fn plonk_load_blind_mont(poly: u32, index: u32) -> Fr {
  let base = ((poly * plonk_params.blind_count) + index) * PLONK_FR_WORDS;
  var z: Fr;
  z.limbs[0] = plonk_blinds[base + 0u];
  z.limbs[1] = plonk_blinds[base + 1u];
  z.limbs[2] = plonk_blinds[base + 2u];
  z.limbs[3] = plonk_blinds[base + 3u];
  z.limbs[4] = plonk_blinds[base + 4u];
  z.limbs[5] = plonk_blinds[base + 5u];
  z.limbs[6] = plonk_blinds[base + 6u];
  z.limbs[7] = plonk_blinds[base + 7u];
  return z;
}

fn plonk_load_scalar_mont(index: u32) -> Fr {
  let base = index * PLONK_FR_WORDS;
  var z: Fr;
  z.limbs[0] = plonk_scalars[base + 0u];
  z.limbs[1] = plonk_scalars[base + 1u];
  z.limbs[2] = plonk_scalars[base + 2u];
  z.limbs[3] = plonk_scalars[base + 3u];
  z.limbs[4] = plonk_scalars[base + 4u];
  z.limbs[5] = plonk_scalars[base + 5u];
  z.limbs[6] = plonk_scalars[base + 6u];
  z.limbs[7] = plonk_scalars[base + 7u];
  return z;
}

fn plonk_store_regular(index: u32, value: Fr) {
  let regular = fr_from_mont(value);
  let base = index * PLONK_FR_WORDS;
  plonk_output[base + 0u] = regular.limbs[0];
  plonk_output[base + 1u] = regular.limbs[1];
  plonk_output[base + 2u] = regular.limbs[2];
  plonk_output[base + 3u] = regular.limbs[3];
  plonk_output[base + 4u] = regular.limbs[4];
  plonk_output[base + 5u] = regular.limbs[5];
  plonk_output[base + 6u] = regular.limbs[6];
  plonk_output[base + 7u] = regular.limbs[7];
}

fn plonk_eval_blind(poly: u32, point: Fr) -> Fr {
  var res = fr_zero();
  var i = plonk_params.blind_count;
  loop {
    if (i == 0u) {
      break;
    }
    i = i - 1u;
    res = fr_add(fr_mul(res, point), plonk_load_blind_mont(poly, i));
  }
  return res;
}

fn plonk_evaluate_quotient(index: u32) -> Fr {
  let twiddle = plonk_load_vector_mont(PLONK_VEC_TWIDDLES, index);
  let next_index = (index + 1u) % plonk_params.count;
  let next_twiddle = plonk_load_vector_mont(PLONK_VEC_TWIDDLES, next_index);

  var l = fr_add(plonk_load_vector_mont(PLONK_VEC_L, index), plonk_eval_blind(PLONK_BLIND_L, twiddle));
  var r = fr_add(plonk_load_vector_mont(PLONK_VEC_R, index), plonk_eval_blind(PLONK_BLIND_R, twiddle));
  var o = fr_add(plonk_load_vector_mont(PLONK_VEC_O, index), plonk_eval_blind(PLONK_BLIND_O, twiddle));
  var z = fr_add(plonk_load_vector_mont(PLONK_VEC_Z, index), plonk_eval_blind(PLONK_BLIND_Z, twiddle));
  let zs = fr_add(plonk_load_vector_mont(PLONK_VEC_Z, next_index), plonk_eval_blind(PLONK_BLIND_Z, next_twiddle));

  var gate = fr_mul(plonk_load_vector_mont(PLONK_VEC_QL, index), l);
  gate = fr_add(gate, fr_mul(plonk_load_vector_mont(PLONK_VEC_QR, index), r));
  gate = fr_add(gate, fr_mul(fr_mul(plonk_load_vector_mont(PLONK_VEC_QM, index), l), r));
  gate = fr_add(gate, fr_mul(plonk_load_vector_mont(PLONK_VEC_QO, index), o));
  gate = fr_add(gate, plonk_load_vector_mont(PLONK_VEC_QK, index));

  let beta = plonk_load_scalar_mont(PLONK_SCALAR_BETA);
  let gamma = plonk_load_scalar_mont(PLONK_SCALAR_GAMMA);
  let alpha = plonk_load_scalar_mont(PLONK_SCALAR_ALPHA);
  let id = fr_mul(fr_mul(twiddle, plonk_load_scalar_mont(PLONK_SCALAR_COSET)), beta);

  var a = fr_add(fr_add(gamma, l), id);
  var b = fr_add(fr_add(fr_mul(id, plonk_load_scalar_mont(PLONK_SCALAR_CS)), r), gamma);
  var c = fr_add(fr_add(fr_mul(id, plonk_load_scalar_mont(PLONK_SCALAR_CSS)), o), gamma);
  let right = fr_mul(fr_mul(fr_mul(a, b), c), z);

  a = fr_add(fr_add(fr_mul(plonk_load_vector_mont(PLONK_VEC_S1, index), beta), l), gamma);
  b = fr_add(fr_add(fr_mul(plonk_load_vector_mont(PLONK_VEC_S2, index), beta), r), gamma);
  c = fr_add(fr_add(fr_mul(plonk_load_vector_mont(PLONK_VEC_S3, index), beta), o), gamma);
  let left = fr_mul(fr_mul(fr_mul(a, b), c), zs);

  let ordering = fr_sub(left, right);
  let lone = fr_mul(plonk_load_scalar_mont(PLONK_SCALAR_LAGRANGE_SCALE), plonk_load_vector_mont(PLONK_VEC_DENOMINATORS, index));
  var local = fr_mul(fr_sub(z, fr_one()), lone);
  local = fr_add(fr_mul(local, alpha), ordering);
  return fr_add(fr_mul(local, alpha), gate);
}

override WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(WORKGROUP_SIZE)
fn fr_plonk_quotient_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= plonk_params.count) {
    return;
  }
  plonk_store_regular(i, plonk_evaluate_quotient(i));
}
