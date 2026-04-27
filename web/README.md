# CurveGPU Web

`curvegpu-web` is the browser-facing TypeScript package for the CurveGPU WebGPU runtime.

The package exposes one public entrypoint:

- [index.ts](/Users/ivo/Work/consensys/repo/webgpu/wnark-crypto/web/index.ts)

The intended usage model is:

1. create one shared WebGPU context
2. create one curve module from that context
3. run `fr`, `fp`, `g1`, `ntt`, `g1msm`, and `g2msm` operations through that module
4. keep smoke tests, benchmarks, and examples outside the package-facing API surface

## Public surface

The exported API includes:

- `createCurveGPUContext`
- `createCurveModule`
- `createBN254`
- `createBLS12381`
- public types for field elements, affine/Jacobian points, NTT, and MSM modules

The browser harnesses live under:

- [web/tests/browser](/Users/ivo/Work/consensys/repo/webgpu/wnark-crypto/web/tests/browser)

The runnable example lives under:

- [web/examples](/Users/ivo/Work/consensys/repo/webgpu/wnark-crypto/web/examples)

Those are validation and documentation tools, not the library entrypoint.

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
} from "curvegpu-web";

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
const msm = await curve.g1msm.pippengerAffine([base, base], [scalarLE(1), scalarLE(1)]);

context.close();
```

## Build

From the repo root:

```sh
make web-build
```

This emits the package entrypoint and declarations under:

- `web/dist/index.js`
- `web/dist/index.d.ts`

The build also emits internal runtime modules used by that entrypoint under `web/dist/src`.
