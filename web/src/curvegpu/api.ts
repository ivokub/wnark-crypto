import type { FieldShape } from "./types.js";

/**
 * Curves currently exposed by the browser library surface.
 */
export type SupportedCurveID = "bn254" | "bls12_381";

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
}

/**
 * Public metadata for the scalar-field NTT module.
 */
export interface NTTModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly field: "fr";
}

/**
 * Public metadata for the G1 MSM module.
 */
export interface MSMModule {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  readonly group: "g1";
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
