/**
 * curvegpu-web — WebGPU-accelerated elliptic curve arithmetic for BN254 and BLS12-381.
 *
 * ## Quick start
 *
 * ```typescript
 * import { createCurveGPUContext, createBN254 } from "curvegpu-web";
 *
 * const ctx = await createCurveGPUContext();
 * const curve = await createBN254(ctx);
 *
 * // G1 scalar multiplication
 * const result = await curve.g1.scalarMul(base, scalar);
 *
 * // Multi-scalar multiplication (Pippenger)
 * const msm = await curve.msm.pippengerPackedJacobianBases(bases, scalars, opts);
 *
 * ctx.close(); // release GPU resources
 * ```
 *
 * ## GPU context
 *
 * `createCurveGPUContext` requests a WebGPU device.  Pass {@link CurveGPUContextOptions}
 * to control power preference, required adapter limits, and debug logging.
 * The returned {@link CurveGPUContext} must be closed with `close()` when no longer needed
 * to release the underlying `GPUDevice` and buffer pool.
 *
 * ## Curve modules
 *
 * Use `createBN254` or `createBLS12381` (or the lower-level `createCurveModule`) to
 * create a {@link CurveModule}.  Each module contains sub-modules for field arithmetic
 * ({@link FieldModule} `fr`, `fp`), curve arithmetic ({@link G1Module}, {@link G2Module}),
 * NTT ({@link NTTModule}), and MSM ({@link MSMModule}, {@link G2MSMModule}).
 *
 * ## Coordinate conventions
 *
 * All field elements and curve points are passed and returned in **Montgomery form** as
 * `Uint32Array` little-endian u32 limb arrays.  Use `splitBytesLEToU32` /
 * `joinU32LimbsToBytesLE` to convert to/from standard byte representations.
 *
 * - **Affine points** — `{x, y}` where each coordinate is a `Uint32Array`.
 * - **Jacobian points** — `{x, y, z}` packed as a flat `Uint32Array` in `[x | y | z]` order.
 * - **G2 affine / Jacobian** — same layout but each coordinate is an Fp2 element
 *   (`{c0, c1}` each a `Uint32Array`).
 *
 * ## Shader bundling
 *
 * By default the library fetches WGSL shader sources at runtime.  To eliminate the runtime
 * `fetch()` dependency, import the generated bundle as a side-effect before creating any
 * curve module:
 *
 * ```typescript
 * import "curvegpu-web/shader_bundle"; // sets bundled shaders via setBundledShaders()
 * ```
 *
 * Or call {@link setBundledShaders} directly with a `Record<string, string>` of path → WGSL.
 *
 * ## Error handling
 *
 * All errors thrown by this library are instances of {@link CurveGPUError} or one of its
 * subclasses:
 * - {@link CurveGPUNotSupportedError} — WebGPU unavailable or required limits not met
 * - {@link CurveGPUDeviceLostError} — GPU device was lost during operation
 * - {@link CurveGPUShaderError} — shader fetch or compilation failure
 *
 * @module
 */
export type {
  CurveGPUAffinePoint,
  CurveGPUAdapterDiagnostics,
  CurveGPUContext,
  CurveGPUContextOptions,
  CurveGPURequestedLimits,
  CurveGPUElementBytes,
  CurveGPUFp2Element,
  CurveGPUG2AffinePoint,
  CurveGPUG2JacobianPoint,
  CurveGPUJacobianPoint,
  CurveGPUPackedPointLayout,
  CurveGPUMSMOptions,
  CurveModule,
  FieldModule,
  G1Module,
  G2Module,
  G2MSMModule,
  MSMModule,
  NTTModule,
  SupportedCurveID,
} from "./curvegpu/api.js";

export {
  CurveGPUError,
  CurveGPUNotSupportedError,
  CurveGPUDeviceLostError,
  CurveGPUShaderError,
} from "./curvegpu/errors.js";

export { setBundledShaders } from "./curvegpu/shaders.js";

export { createCurveGPUContext } from "./curvegpu/context.js";

export {
  createBLS12381,
  createBN254,
  createCurveModule,
  curveDefinition,
  supportedCurveIds,
} from "./curvegpu/curves.js";

export type { CurveDefinition } from "./curvegpu/curves.js";

export type { CurveID, FieldID, FieldShape } from "./curvegpu/types.js";
export { shapeFor } from "./curvegpu/types.js";

export {
  joinU32LimbsToBigUint64,
  joinU32LimbsToBytesLE,
  splitBigUint64WordsToU32,
  splitBytesLEToU32,
} from "./curvegpu/convert.js";
