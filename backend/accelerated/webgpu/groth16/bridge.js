import { createBLS12377, createBLS12381, createBN254, createCurveGPUContext } from "/web/dist/index.js";

const CURVE_CONFIG = {
  bn254: {
    g1CoordinateBytes: 32,
    g1PointBytes: 96,
    g2ComponentBytes: 32,
    g2PointBytes: 192,
  },
  bls12_381: {
    g1CoordinateBytes: 48,
    g1PointBytes: 144,
    g2ComponentBytes: 48,
    g2PointBytes: 288,
  },
  bls12_377: {
    g1CoordinateBytes: 48,
    g1PointBytes: 144,
    g2ComponentBytes: 48,
    g2PointBytes: 288,
  },
};

let contextPromise = null;
const modulePromises = new Map();
const keyCache = new Map();
let nextHandle = 1;

function cloneBytes(bytes) {
  return new Uint8Array(bytes);
}

function createCurveForID(curve, context) {
  switch (curve) {
    case "bn254":
      return createBN254(context);
    case "bls12_381":
      return createBLS12381(context);
    case "bls12_377":
      return createBLS12377(context);
    default:
      throw new Error(`unsupported curve ${curve}`);
  }
}

async function getContext() {
  if (!contextPromise) {
    contextPromise = createCurveGPUContext();
  }
  return contextPromise;
}

async function getCurveModule(curve) {
  if (!modulePromises.has(curve)) {
    modulePromises.set(
      curve,
      (async () => {
        const context = await getContext();
        return createCurveForID(curve, context);
      })(),
    );
  }
  return modulePromises.get(curve);
}

function unpackG1JacobianPoint(curve, packedPoint) {
  const coordinateBytes = CURVE_CONFIG[curve].g1CoordinateBytes;
  return {
    x: cloneBytes(packedPoint.slice(0, coordinateBytes)),
    y: cloneBytes(packedPoint.slice(coordinateBytes, 2 * coordinateBytes)),
    z: cloneBytes(packedPoint.slice(2 * coordinateBytes, 3 * coordinateBytes)),
  };
}

function unpackG2JacobianPoint(curve, packedPoint) {
  const componentBytes = CURVE_CONFIG[curve].g2ComponentBytes;
  return {
    x: {
      c0: cloneBytes(packedPoint.slice(0, componentBytes)),
      c1: cloneBytes(packedPoint.slice(componentBytes, 2 * componentBytes)),
    },
    y: {
      c0: cloneBytes(packedPoint.slice(2 * componentBytes, 3 * componentBytes)),
      c1: cloneBytes(packedPoint.slice(3 * componentBytes, 4 * componentBytes)),
    },
    z: {
      c0: cloneBytes(packedPoint.slice(4 * componentBytes, 5 * componentBytes)),
      c1: cloneBytes(packedPoint.slice(5 * componentBytes, 6 * componentBytes)),
    },
  };
}

function getKey(handle) {
  const entry = keyCache.get(handle);
  if (!entry) {
    throw new Error(`unknown Groth16 key handle ${handle}`);
  }
  return entry;
}

async function init(curve) {
  const [context] = await Promise.all([getContext(), getCurveModule(curve)]);
  return {
    curve,
    adapter: {
      vendor: context.diagnostics.vendor ?? "",
      architecture: context.diagnostics.architecture ?? "",
      description: context.diagnostics.description ?? "",
    },
  };
}

async function prepareKey(curve, payload) {
  await init(curve);
  const handle = `${curve}:${nextHandle++}`;
  keyCache.set(handle, {
    curve,
    g1A: cloneBytes(payload.g1A),
    g1ACount: Number(payload.g1ACount),
    g1B: cloneBytes(payload.g1B),
    g1BCount: Number(payload.g1BCount),
    g1K: cloneBytes(payload.g1K),
    g1KCount: Number(payload.g1KCount),
    g1Z: cloneBytes(payload.g1Z),
    g1ZCount: Number(payload.g1ZCount),
    g2B: cloneBytes(payload.g2B),
    g2BCount: Number(payload.g2BCount),
  });
  return { handle };
}

async function releaseKey(handle) {
  keyCache.delete(handle);
}

async function msmG1(handle, vectorName, scalarsPacked) {
  const entry = getKey(handle);
  const module = await getCurveModule(entry.curve);
  const config = CURVE_CONFIG[entry.curve];
  const basesPacked = entry[vectorName];
  const count = entry[`${vectorName}Count`];
  if (!(basesPacked instanceof Uint8Array) || typeof count !== "number") {
    throw new Error(`missing cached G1 vector ${vectorName}`);
  }
  const resultPacked = await module.g1msm.pippengerPackedJacobianBases(basesPacked, cloneBytes(scalarsPacked), {
    count: 1,
    termsPerInstance: count,
    window: module.g1msm.bestWindow(count),
  });
  const jacobian = unpackG1JacobianPoint(entry.curve, resultPacked.slice(0, config.g1PointBytes));
  const affine = await module.g1.jacobianToAffine(jacobian);
  const out = new Uint8Array(2 * config.g1CoordinateBytes);
  out.set(affine.x, 0);
  out.set(affine.y, config.g1CoordinateBytes);
  return out;
}

async function msmG2(handle, vectorName, scalarsPacked) {
  const entry = getKey(handle);
  const module = await getCurveModule(entry.curve);
  const config = CURVE_CONFIG[entry.curve];
  const basesPacked = entry[vectorName];
  const count = entry[`${vectorName}Count`];
  if (!(basesPacked instanceof Uint8Array) || typeof count !== "number") {
    throw new Error(`missing cached G2 vector ${vectorName}`);
  }
  const resultPacked = await module.g2msm.pippengerPackedJacobianBases(basesPacked, cloneBytes(scalarsPacked), {
    count: 1,
    termsPerInstance: count,
    window: module.g2msm.bestWindow(count),
  });
  const jacobian = unpackG2JacobianPoint(entry.curve, resultPacked.slice(0, config.g2PointBytes));
  const affine = await module.g2.jacobianToAffine(jacobian);
  const out = new Uint8Array(4 * config.g2ComponentBytes);
  out.set(affine.x.c0, 0);
  out.set(affine.x.c1, config.g2ComponentBytes);
  out.set(affine.y.c0, 2 * config.g2ComponentBytes);
  out.set(affine.y.c1, 3 * config.g2ComponentBytes);
  return out;
}

async function computeH(curve, aPacked, bPacked, cPacked) {
  const module = await getCurveModule(curve);
  return module.ntt.computeGroth16QuotientPackedRegular(cloneBytes(aPacked), cloneBytes(bPacked), cloneBytes(cPacked));
}

globalThis.wnarkGroth16WebGPU = {
  init,
  prepareKey,
  releaseKey,
  msmG1,
  msmG2,
  computeH,
};
