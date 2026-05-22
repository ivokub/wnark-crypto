import type { CurveGPUContext, G1Module, G1MSMModule, SupportedCurveID } from "./api.js";

const CURVE_CONFIG: Record<SupportedCurveID, {
  g1CoordinateBytes: number;
  g1PointBytes: number;
}> = {
  bn254: {
    g1CoordinateBytes: 32,
    g1PointBytes: 96,
  },
  bls12_381: {
    g1CoordinateBytes: 48,
    g1PointBytes: 144,
  },
  bls12_377: {
    g1CoordinateBytes: 48,
    g1PointBytes: 144,
  },
};

type BridgeDependencies = {
  context: CurveGPUContext;
  curve: SupportedCurveID;
  g1: G1Module;
  g1msm: G1MSMModule;
};

type CachedKey = {
  curve: SupportedCurveID;
  kzgLagrange: Uint8Array;
  kzgLagrangeCount: number;
};

let activeBridge: BridgeDependencies | null = null;
let nextHandle = 1;
const keyCache = new Map<string, CachedKey>();

function cloneBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes);
}

function assertBridge(curve: string): BridgeDependencies {
  if (!activeBridge) {
    throw new Error("PLONK WebGPU bridge is not initialized");
  }
  if (curve !== activeBridge.curve) {
    throw new Error(`PLONK WebGPU bridge is bound to ${activeBridge.curve}, got ${curve}`);
  }
  return activeBridge;
}

function getKey(handle: string): CachedKey {
  const entry = keyCache.get(handle);
  if (!entry) {
    throw new Error(`unknown PLONK key handle ${handle}`);
  }
  return entry;
}

function unpackG1JacobianPoint(curve: SupportedCurveID, packedPoint: Uint8Array) {
  const coordinateBytes = CURVE_CONFIG[curve].g1CoordinateBytes;
  return {
    x: cloneBytes(packedPoint.slice(0, coordinateBytes)),
    y: cloneBytes(packedPoint.slice(coordinateBytes, 2 * coordinateBytes)),
    z: cloneBytes(packedPoint.slice(2 * coordinateBytes, 3 * coordinateBytes)),
  };
}

async function init(curve: SupportedCurveID) {
  const bridge = assertBridge(curve);
  return {
    curve,
    adapter: {
      vendor: bridge.context.diagnostics.vendor ?? "",
      architecture: bridge.context.diagnostics.architecture ?? "",
      description: bridge.context.diagnostics.description ?? "",
    },
  };
}

async function prepareKey(curve: SupportedCurveID, payload: Record<string, Uint8Array | number | undefined>) {
  assertBridge(curve);
  const kzgLagrange = payload.kzgLagrange;
  const kzgLagrangeCount = Number(payload.kzgLagrangeCount);
  if (!(kzgLagrange instanceof Uint8Array)) {
    throw new Error("PLONK key payload is missing kzgLagrange");
  }
  if (!Number.isInteger(kzgLagrangeCount) || kzgLagrangeCount <= 0) {
    throw new Error(`invalid PLONK kzgLagrangeCount ${payload.kzgLagrangeCount}`);
  }
  const expectedBytes = kzgLagrangeCount * CURVE_CONFIG[curve].g1PointBytes;
  if (kzgLagrange.byteLength !== expectedBytes) {
    throw new Error(`PLONK kzgLagrange expected ${expectedBytes} bytes, got ${kzgLagrange.byteLength}`);
  }
  const handle = `${curve}:${nextHandle++}`;
  const entry: CachedKey = {
    curve,
    kzgLagrange: cloneBytes(kzgLagrange),
    kzgLagrangeCount,
  };
  keyCache.set(handle, entry);
  return { handle };
}

async function msmG1(handle: string, vectorName: string, scalarsPacked: Uint8Array, start = 0, count?: number) {
  const entry = getKey(handle);
  const bridge = assertBridge(entry.curve);
  const config = CURVE_CONFIG[entry.curve];
  if (vectorName !== "kzgLagrange") {
    throw new Error(`missing cached PLONK G1 vector ${vectorName}`);
  }
  const termCount = count ?? entry.kzgLagrangeCount - start;
  if (!Number.isInteger(start) || start < 0 || !Number.isInteger(termCount) || termCount <= 0) {
    throw new Error(`invalid PLONK MSM range start=${start} count=${termCount}`);
  }
  if (start + termCount > entry.kzgLagrangeCount) {
    throw new Error(`PLONK MSM range exceeds kzgLagrange: start=${start} count=${termCount}`);
  }
  const baseStart = start * config.g1PointBytes;
  const baseEnd = (start + termCount) * config.g1PointBytes;
  const basesPacked = entry.kzgLagrange.subarray(baseStart, baseEnd);
  const resultPacked = await bridge.g1msm.pippengerPackedJacobianBases(basesPacked, cloneBytes(scalarsPacked), {
    count: 1,
    termsPerInstance: termCount,
    window: bridge.g1msm.bestWindow(termCount),
  });
  const jacobian = unpackG1JacobianPoint(entry.curve, resultPacked.slice(0, config.g1PointBytes));
  const affine = await bridge.g1.jacobianToAffine(jacobian);
  const out = new Uint8Array(2 * config.g1CoordinateBytes);
  out.set(affine.x, 0);
  out.set(affine.y, config.g1CoordinateBytes);
  return out;
}

export function installPlonkWebGPUBridge(dependencies: BridgeDependencies): void {
  activeBridge = dependencies;
  (globalThis as typeof globalThis & { wnarkPlonkWebGPU?: unknown }).wnarkPlonkWebGPU = {
    init,
    prepareKey,
    msmG1,
  };
}
