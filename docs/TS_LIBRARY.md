# CurveGPU Browser Library

`web/src/index.ts` is the public TypeScript entrypoint for the browser-side CurveGPU library. The compiled browser module is emitted to `web/dist/index.js`.

The intended usage model is:

1. create one shared WebGPU context
2. create one curve module from that context
3. call `fr`, `fp`, `g1`, `ntt`, and `msm` operations through that module
4. keep smoke tests and benchmarks under the browser test harness, not in the library entrypoint

## Byte conventions

- `fr` and `fp` methods operate on little-endian Montgomery-form byte strings unless you explicitly call `toMontgomery` or `fromMontgomery`
- G1 affine and Jacobian coordinates use the same field-byte representation as the selected curve's fixtures and shader interfaces
- scalar-multiplication and MSM scalars are 32-byte little-endian scalar-field elements in regular form

## Minimal usage

```ts
import {
  createCurveGPUContext,
  createBN254,
  type CurveGPUAffinePoint,
} from "./index.js";

function scalarLE(value: number): Uint8Array {
  const out = new Uint8Array(32);
  out[0] = value;
  return out;
}

const context = await createCurveGPUContext();
const curve = createBN254(context);

const one = await curve.fr.montOne();
const two = await curve.fr.add(one, one);
const twoRegular = await curve.fr.fromMontgomery(two);

const base: CurveGPUAffinePoint = {
  x: /* affine x bytes */,
  y: /* affine y bytes */,
};

const doubled = await curve.g1.scalarMulAffine(base, scalarLE(2));
const msm = await curve.msm.pippengerAffine([base, base], [scalarLE(1), scalarLE(1)]);

context.close();
```

## Runnable example

There is one small consumer-oriented browser example page:

- `web/examples/library_example.html?curve=bn254`
- `web/examples/library_example.html?curve=bls12_381`

That example uses only the public library entrypoint and demonstrates:

- field arithmetic
- NTT round-trip
- G1 scalar multiplication
- G1 MSM

## Test and benchmark harnesses

Smoke tests and benchmarks stay under the unified browser harness page:

- `web/tests/browser/curvegpu.html`

Those pages are harnesses for validation and performance work. They are not the intended production entrypoint for consumers of the library.

For MSM benchmarks, fixed G1 base fixtures live under `testdata/fixtures/g1`. The checked-in BLS12-381 fixture is used automatically, and larger local fixtures can be generated with:

- `make fixture-bn254-g1 COUNT=<points>`
- `make fixture-bls12_381-g1 COUNT=<points>`
