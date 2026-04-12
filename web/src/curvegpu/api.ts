import type { FieldShape } from "./types.js";

/**
 * Curves currently exposed by the browser library surface.
 */
export type SupportedCurveID = "bn254" | "bls12_381";

/**
 * Canonical byte representation for field and scalar values.
 */
export type CurveGPUElementBytes = Uint8Array;

/**
 * Affine G1 point represented as little-endian field-element byte strings.
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
  close(): void;
}

/**
 * Public metadata for a field module.
 *
 * Operational methods are added on top of this stable boundary in follow-up
 * refactor steps.
 */
export interface FieldModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly field: "fr" | "fp";
  readonly shape: FieldShape;
  readonly byteSize: number;
  zero(): CurveGPUElementBytes;
  copy(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  copyBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  one(): Promise<CurveGPUElementBytes>;
  equal(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<boolean>;
  equalBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<boolean[]>;
  add(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  addBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  sub(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  subBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  neg(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  negBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  double(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  doubleBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  mul(a: CurveGPUElementBytes, b: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  mulBatch(a: readonly CurveGPUElementBytes[], b: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  square(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  squareBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  normalize(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  normalizeBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  toMont(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  toMontBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  fromMont(value: CurveGPUElementBytes): Promise<CurveGPUElementBytes>;
  fromMontBatch(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
}

/**
 * Public metadata for the G1 module.
 *
 * Operational methods are added on top of this stable boundary in follow-up
 * refactor steps.
 */
export interface G1Module {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly coordinateBytes: number;
  readonly pointBytes: number;
  readonly zeroHex: string;
  affineInfinity(): CurveGPUAffinePoint;
  jacobianZero(): CurveGPUJacobianPoint;
  copy(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  copyBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  jacobianInfinity(): Promise<CurveGPUJacobianPoint>;
  jacobianInfinityBatch(count: number): Promise<CurveGPUJacobianPoint[]>;
  affineToJacobian(point: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  affineToJacobianBatch(points: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  negJacobian(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  negJacobianBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  doubleJacobian(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  doubleJacobianBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  addMixed(point: CurveGPUJacobianPoint, affine: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  addMixedBatch(points: readonly CurveGPUJacobianPoint[], affine: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  jacobianToAffine(point: CurveGPUJacobianPoint): Promise<CurveGPUJacobianPoint>;
  jacobianToAffineBatch(points: readonly CurveGPUJacobianPoint[]): Promise<CurveGPUJacobianPoint[]>;
  affineAdd(a: CurveGPUAffinePoint, b: CurveGPUAffinePoint): Promise<CurveGPUJacobianPoint>;
  affineAddBatch(a: readonly CurveGPUAffinePoint[], b: readonly CurveGPUAffinePoint[]): Promise<CurveGPUJacobianPoint[]>;
  scalarMulAffine(base: CurveGPUAffinePoint, scalar: CurveGPUElementBytes): Promise<CurveGPUJacobianPoint>;
  scalarMulAffineBatch(bases: readonly CurveGPUAffinePoint[], scalars: readonly CurveGPUElementBytes[]): Promise<CurveGPUJacobianPoint[]>;
}

/**
 * Public metadata for the scalar-field NTT module.
 */
export interface NTTModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly field: "fr";
  supportedSizes(): Promise<number[]>;
  forward(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
  inverse(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]>;
}

/**
 * Public metadata for the G1 MSM module.
 */
export interface MSMModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly group: "g1";
  bestWindow(termCount: number): number;
  pippengerAffine(
    bases: readonly CurveGPUAffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
    options?: CurveGPUMSMOptions,
  ): Promise<CurveGPUJacobianPoint>;
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
