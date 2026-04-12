import type { FieldShape } from "./types.js";

/**
 * Curves currently exposed by the browser library surface.
 */
export type SupportedCurveID = "bn254" | "bls12_381";

/**
 * Canonical byte representation for field and scalar values.
 *
 * For `fr` and `fp` module operations, values are little-endian byte strings
 * in Montgomery form unless explicitly converted with
 * `toMontgomery` / `fromMontgomery`.
 *
 * For G1 scalar multiplication and MSM, scalars are little-endian 32-byte
 * scalar-field elements in regular form.
 */
export type CurveGPUElementBytes = Uint8Array;

/**
 * Affine G1 point represented as little-endian field-element byte strings.
 *
 * Coordinates use the same field representation as the curve fixtures and
 * shader interfaces for the selected curve.
 */
export interface CurveGPUAffinePoint {
  x: Uint8Array;
  y: Uint8Array;
}

/**
 * Jacobian G1 point represented as little-endian field-element byte strings.
 */
export interface CurveGPUJacobianPoint {
  x: Uint8Array;
  y: Uint8Array;
  z: Uint8Array;
}

/**
 * Options for affine MSM execution.
 */
export type CurveGPUMSMOptions = {
  count?: number;
  termsPerInstance?: number;
  window?: number;
  maxChunkSize?: number;
};

/**
 * Subset of device limits that matter for the current curve workloads.
 */
export type CurveGPURequestedLimits = {
  maxStorageBufferBindingSize?: number;
  maxBufferSize?: number;
};

/**
 * Human-readable adapter details useful for logging, debugging, and telemetry.
 */
export type CurveGPUAdapterDiagnostics = {
  vendor?: string;
  architecture?: string;
  description?: string;
  isFallbackAdapter?: boolean;
};

/**
 * Options for acquiring a browser WebGPU context.
 */
export type CurveGPUContextOptions = {
  powerPreference?: GPUPowerPreference;
  requireAdapterLimits?: boolean;
  requiredLimits?: CurveGPURequestedLimits;
};

/**
 * Shared browser WebGPU context for all curve operations.
 *
 * This is the top-level object a consumer creates once, then reuses for
 * field, group, NTT, and MSM work.
 */
export interface CurveGPUContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly adapterInfo: GPUAdapterInfo | null;
  readonly diagnostics: CurveGPUAdapterDiagnostics;
  readonly requestedLimits: CurveGPURequestedLimits;
  /**
   * Release any library-owned resources associated with the context.
   *
   * Browser WebGPU device lifetime is still managed by the browser, so this is
   * currently a logical shutdown hook rather than a hard device destroy.
   */
  close(): void;
}

/**
 * Field arithmetic bound to a specific curve field.
 *
 * All methods except `toMontgomery` and `fromMontgomery` operate on
 * Montgomery-form little-endian byte strings.
 *
 * Batch variants execute the same operation element-wise over equal-length
 * slices.
 */
export interface FieldModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly field: "fr" | "fp";
  readonly shape: FieldShape;
  readonly byteSize: number;
  /** Return the additive identity as a zero-filled byte string. */
  zero(): CurveGPUElementBytes;
  /** Copy one element through the GPU implementation. */
  copy(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  copyBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Return the multiplicative identity in Montgomery form. */
  montOne(): Promise<CurveGPUElementBytes>;
  /** Check modular equality. */
  equal(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<boolean>;
  equalBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<boolean[]>;
  /** Modular addition. */
  add(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  addBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Modular subtraction. */
  sub(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  subBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Modular negation. */
  neg(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  negBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Modular doubling. */
  double(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  doubleBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Modular multiplication. */
  mul(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  mulBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Modular squaring. */
  square(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  squareBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Reduce a value into canonical Montgomery form. */
  normalizeMont(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  normalizeMontBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Convert a regular little-endian field element into Montgomery form. */
  toMontgomery(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  toMontgomeryBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Convert a Montgomery-form element back into regular little-endian bytes. */
  fromMontgomery(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  fromMontgomeryBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
}

/**
 * G1 point operations for a specific curve.
 *
 * Affine inputs are passed as `x` and `y` byte strings. Jacobian outputs use
 * three coordinates in the same field representation as the selected curve.
 */
export interface G1Module {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly coordinateBytes: number;
  readonly pointBytes: number;
  readonly zeroHex: string;
  /** Return the affine point at infinity. */
  affineInfinity(): CurveGPUAffinePoint;
  /** Return the zero Jacobian point. */
  jacobianZero(): CurveGPUJacobianPoint;
  /** Copy a Jacobian point through the GPU implementation. */
  copy(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  copyBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  /** Construct the Jacobian point at infinity. */
  jacobianInfinity(): Promise<CurveGPUJacobianPoint>;
  jacobianInfinityBatch(count: number): Promise<CurveGPUJacobianPoint[]>;
  /** Lift affine points into Jacobian coordinates. */
  affineToJacobian(point: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  affineToJacobianBatch(points: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  /** Negate Jacobian points. */
  negJacobian(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  negJacobianBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  /** Double Jacobian points. */
  doubleJacobian(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  doubleJacobianBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  /** Add an affine point into a Jacobian accumulator. */
  addMixed(point: CurveGPUJacobianPoint, affine: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  addMixedBatch(points: readonly CurveGPUJacobianPoint[], affine: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  /**
   * Convert Jacobian points to affine coordinates.
   *
   * The returned object keeps the `z` field for compatibility with existing
   * fixtures; consumers that need a strict affine point should use `x` and `y`.
   */
  jacobianToAffine(point: CurveGPUJacobianPoint): Promise<CurveGPUAffinePoint>;
  jacobianToAffineBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUAffinePoint[]>;
  /** Add two affine points and return the result in Jacobian form. */
  affineAdd(a: CurveGPUAffinePoint, b: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  affineAddBatch(a: readonly CurveGPUAffinePoint[], b: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  /** Multiply affine bases by scalar-field elements. */
  scalarMulAffine(base: CurveGPUAffinePoint, scalar: CurveGPUElementBytes): Promise<CurveGPUJacobianPoint>;
  scalarMulAffineBatch(bases: readonly CurveGPUAffinePoint[], scalars: readonly CurveGPUElementBytes[]): Promise<CurveGPUJacobianPoint[]>;
}

/**
 * Scalar-field NTT module for a specific curve.
 */
export interface NTTModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly field: "fr";
  /** Report the power-of-two domain sizes available from loaded metadata. */
  supportedSizes(): Promise<number[]>;
  /** Run the forward NTT over a power-of-two batch of Montgomery-form values. */
  forward(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  /** Run the inverse NTT over a power-of-two batch of Montgomery-form values. */
  inverse(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
}

/**
 * Multi-scalar multiplication module over G1 affine bases.
 */
export interface MSMModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly group: "g1";
  /** Choose the default Pippenger window for a given term count. */
  bestWindow(termCount: number): number;
  /** Run a single affine-base Pippenger MSM and return the result in Jacobian form. */
  pippengerAffine(
    bases: readonly CurveGPUAffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
    options?: CurveGPUMSMOptions,
  ): Promise<CurveGPUJacobianPoint>;
  /** Run a batched affine-base Pippenger MSM. */
  pippengerAffineBatch(
    bases: readonly CurveGPUAffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
    options: CurveGPUMSMOptions,
  ): Promise<CurveGPUJacobianPoint[]>;
}

/**
 * High-level curve module returned by the library.
 *
 * This groups the curve-specific submodules behind one stable object per
 * supported curve.
 */
export interface CurveModule {
  readonly id: SupportedCurveID;
  readonly context: CurveGPUContext;
  readonly fr: FieldModule;
  readonly fp: FieldModule;
  readonly g1: G1Module;
  readonly ntt: NTTModule;
  readonly msm: MSMModule;
}
