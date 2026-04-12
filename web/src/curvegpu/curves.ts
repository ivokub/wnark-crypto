import type { CurveModule, CurveGPUContext, SupportedCurveID } from "./api.js";
import { createFieldModule } from "./field_module.js";
import { createG1Module } from "./g1_module.js";
import { createMSMModule } from "./msm_module.js";
import { createNTTModule } from "./ntt_module.js";
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
  readonly frNTTDomainPath?: string;
  readonly frModulusHex?: string;
  readonly fpArithShaderPath: string;
  readonly g1ArithShaderParts: readonly string[];
  readonly g1MSMShaderParts: readonly string[];
  readonly pippengerMode: "simple" | "weighted";
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
    frNTTDomainPath: "/testdata/vectors/fr/bn254_ntt_domains.json",
    frModulusHex: "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
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
    pippengerMode: "simple",
    coordinateBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
  },
  bls12_381: {
    id: "bls12_381",
    frArithShaderPath: "/shaders/curves/bls12_381/fr_arith.wgsl",
    frVectorShaderPath: "/shaders/curves/bls12_381/fr_vector.wgsl",
    frNTTShaderPath: "/shaders/curves/bls12_381/fr_ntt.wgsl",
    frNTTDomainPath: "/testdata/vectors/fr/bls12_381_ntt_domains.json",
    frModulusHex: "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
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
    pippengerMode: "weighted",
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

/**
 * Create the high-level curve module for a supported curve.
 *
 * This establishes the stable public object shape that later steps populate
 * with concrete field, NTT, group, and MSM operations.
 */
export function createCurveModule(context: CurveGPUContext, curve: SupportedCurveID): CurveModule {
  const definition = curveDefinition(curve);
  const frShape = shapeFor(curve, "fr");
  const fpShape = shapeFor(curve, "fp");
  const fr = createFieldModule(context, curve, "fr", {
    byteSize: frShape.byteSize,
    shaderPath: definition.frArithShaderPath,
    entryPoint: "fr_ops_main",
    label: `${curve}-fr`,
    shape: frShape,
  });
  const fp = createFieldModule(context, curve, "fp", {
    byteSize: fpShape.byteSize,
    shaderPath: definition.fpArithShaderPath,
    entryPoint: "fp_ops_main",
    label: `${curve}-fp`,
    shape: fpShape,
  });
  return {
    id: curve,
    context,
    fr,
    fp,
    g1: createG1Module(
      context,
      {
        curve: definition.id,
        coordinateBytes: definition.coordinateBytes,
        pointBytes: definition.pointBytes,
        zeroHex: definition.zeroHex,
        shaderParts: definition.g1ArithShaderParts,
      },
      fp,
    ),
    ntt: createNTTModule(
      context,
      {
        curve: definition.id,
        vectorShaderPath: definition.frVectorShaderPath,
        nttShaderPath: definition.frNTTShaderPath,
        domainPath: definition.frNTTDomainPath ?? "",
        modulusHex: definition.frModulusHex ?? "",
      },
      fr,
    ),
    msm: createMSMModule(
      context,
      {
        curve: definition.id,
        coordinateBytes: definition.coordinateBytes,
        pointBytes: definition.pointBytes,
        shaderParts: definition.g1MSMShaderParts,
        mode: definition.pippengerMode,
      },
      fp,
    ),
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
