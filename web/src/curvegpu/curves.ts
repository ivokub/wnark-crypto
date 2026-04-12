import type { CurveModule, CurveGPUContext, FieldModule, G1Module, MSMModule, NTTModule, SupportedCurveID } from "./api.js";
import { shapeFor } from "./types.js";

/**
 * Runtime metadata for a supported curve.
 *
 * This is kept separate from the page harnesses so the library can evolve
 * around one shared source of curve-specific facts.
 */
export interface CurveDefinition {
  readonly id: SupportedCurveID;
  readonly frArithShaderPath: string;
  readonly frVectorShaderPath: string;
  readonly frNTTShaderPath: string;
  readonly fpArithShaderPath: string;
  readonly g1ArithShaderParts: readonly string[];
  readonly g1MSMShaderParts: readonly string[];
  readonly coordinateBytes: number;
  readonly pointBytes: number;
  readonly zeroHex: string;
}

const CURVE_DEFINITIONS: Record<SupportedCurveID, CurveDefinition> = {
  bn254: {
    id: "bn254",
    frArithShaderPath: "/shaders/curves/bn254/fr_arith.wgsl",
    frVectorShaderPath: "/shaders/curves/bn254/fr_vector.wgsl",
    frNTTShaderPath: "/shaders/curves/bn254/fr_ntt.wgsl",
    fpArithShaderPath: "/shaders/curves/bn254/fp_arith.wgsl",
    g1ArithShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl",
    ],
    g1MSMShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl",
    ],
    coordinateBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
  },
  bls12_381: {
    id: "bls12_381",
    frArithShaderPath: "/shaders/curves/bls12_381/fr_arith.wgsl",
    frVectorShaderPath: "/shaders/curves/bls12_381/fr_vector.wgsl",
    frNTTShaderPath: "/shaders/curves/bls12_381/fr_ntt.wgsl",
    fpArithShaderPath: "/shaders/curves/bls12_381/fp_arith.wgsl",
    g1ArithShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-core",
      "/shaders/curves/bls12_381/g1_arith.wgsl",
    ],
    g1MSMShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-core",
      "/shaders/curves/bls12_381/g1_msm.wgsl",
    ],
    coordinateBytes: 48,
    pointBytes: 144,
    zeroHex:
      "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  },
};

/**
 * Ordered list of curves currently exposed by the browser library.
 */
export const supportedCurveIds = Object.freeze(Object.keys(CURVE_DEFINITIONS)) as readonly SupportedCurveID[];

/**
 * Return the runtime metadata for a supported curve.
 */
export function curveDefinition(curve: SupportedCurveID): CurveDefinition {
  return CURVE_DEFINITIONS[curve];
}

function createFieldModule(context: CurveGPUContext, curve: SupportedCurveID, field: "fr" | "fp"): FieldModule {
  const shape = shapeFor(curve, field);
  return {
    context,
    curve,
    field,
    shape,
    byteSize: shape.byteSize,
  };
}

function createG1Module(context: CurveGPUContext, definition: CurveDefinition): G1Module {
  return {
    context,
    curve: definition.id,
    coordinateBytes: definition.coordinateBytes,
    pointBytes: definition.pointBytes,
    zeroHex: definition.zeroHex,
  };
}

function createNTTModule(context: CurveGPUContext, curve: SupportedCurveID): NTTModule {
  return {
    context,
    curve,
    field: "fr",
  };
}

function createMSMModule(context: CurveGPUContext, curve: SupportedCurveID): MSMModule {
  return {
    context,
    curve,
    group: "g1",
  };
}

/**
 * Create the high-level curve module for a supported curve.
 *
 * This establishes the stable public object shape that later steps populate
 * with concrete field, NTT, group, and MSM operations.
 */
export function createCurveModule(context: CurveGPUContext, curve: SupportedCurveID): CurveModule {
  const definition = curveDefinition(curve);
  return {
    id: curve,
    context,
    fr: createFieldModule(context, curve, "fr"),
    fp: createFieldModule(context, curve, "fp"),
    g1: createG1Module(context, definition),
    ntt: createNTTModule(context, curve),
    msm: createMSMModule(context, curve),
  };
}

/**
 * Create the BN254 module bound to an existing context.
 */
export function createBN254(context: CurveGPUContext): CurveModule {
  return createCurveModule(context, "bn254");
}

/**
 * Create the BLS12-381 module bound to an existing context.
 */
export function createBLS12381(context: CurveGPUContext): CurveModule {
  return createCurveModule(context, "bls12_381");
}
