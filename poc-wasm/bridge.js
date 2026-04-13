import { createCurveGPUContext, createCurveModule } from "/web/dist/index.js";

const query = new URLSearchParams(window.location.search);
const fixtureRev = query.get("fixture-rev") ?? query.get("fixtureRev") ?? "1";

const CURVE_CONFIG = {
  bn254: {
    frBytes: 32,
    coordinateBytes: 32,
    pointBytes: 96,
    fixtureBinPath: `/testdata/fixtures/g1/bn254_bases_jacobian.bin?v=${encodeURIComponent(fixtureRev)}`,
    fixtureJSONPath: `/testdata/fixtures/g1/bn254_bases_jacobian.json?v=${encodeURIComponent(fixtureRev)}`,
  },
  bls12_381: {
    frBytes: 32,
    coordinateBytes: 48,
    pointBytes: 144,
    fixtureBinPath: `/testdata/fixtures/g1/bls12_381_bases_jacobian.bin?v=${encodeURIComponent(fixtureRev)}`,
    fixtureJSONPath: `/testdata/fixtures/g1/bls12_381_bases_jacobian.json?v=${encodeURIComponent(fixtureRev)}`,
  },
};

let contextPromise = null;
const modulePromises = new Map();
const fixturePromises = new Map();

function cloneBytes(bytes) {
  return new Uint8Array(bytes);
}

function hex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return hex(new Uint8Array(digest));
}

function concatBytes(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function generateRegularLEPacked(count, byteSize, start = 1) {
  const out = new Uint8Array(count * byteSize);
  for (let i = 0; i < count; i += 1) {
    let value = BigInt(start + i);
    const offset = i * byteSize;
    for (let j = 0; j < byteSize; j += 1) {
      out[offset + j] = Number(value & 0xffn);
      value >>= 8n;
    }
  }
  return out;
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
        return createCurveModule(context, curve);
      })(),
    );
  }
  return modulePromises.get(curve);
}

async function getFixture(curve) {
  if (!fixturePromises.has(curve)) {
    const config = CURVE_CONFIG[curve];
    fixturePromises.set(
      curve,
      (async () => {
        const [metaResp, binResp] = await Promise.all([
          fetch(config.fixtureJSONPath),
          fetch(config.fixtureBinPath),
        ]);
        if (!metaResp.ok) {
          throw new Error(`failed to fetch fixture metadata: ${metaResp.status}`);
        }
        if (!binResp.ok) {
          throw new Error(`failed to fetch fixture binary: ${binResp.status}`);
        }
        const meta = await metaResp.json();
        const bytes = new Uint8Array(await binResp.arrayBuffer());
        return { meta, bytes };
      })(),
    );
  }
  return fixturePromises.get(curve);
}

async function init(curve) {
  const started = performance.now();
  const [context, module, fixture] = await Promise.all([
    getContext(),
    getCurveModule(curve),
    getFixture(curve),
  ]);
  return {
    curve,
    initMs: performance.now() - started,
    adapter: {
      vendor: context.diagnostics.vendor ?? "",
      architecture: context.diagnostics.architecture ?? "",
      description: context.diagnostics.description ?? "",
    },
    fixturePoints: fixture.meta.count,
    strategy: module.msm.bestWindow(1 << 12),
  };
}

function unpackJacobianPoint(curve, packedPoint) {
  const coordinateBytes = CURVE_CONFIG[curve].coordinateBytes;
  return {
    x: cloneBytes(packedPoint.slice(0, coordinateBytes)),
    y: cloneBytes(packedPoint.slice(coordinateBytes, 2 * coordinateBytes)),
    z: cloneBytes(packedPoint.slice(2 * coordinateBytes, 3 * coordinateBytes)),
  };
}

async function runNtt(curve, size, start = 1) {
  const config = CURVE_CONFIG[curve];
  const module = await getCurveModule(curve);
  const inputsPacked = generateRegularLEPacked(size, config.frBytes, start);
  const started = performance.now();
  const outputRegular = await module.ntt.forwardPackedRegular(inputsPacked);
  const durationMs = performance.now() - started;
  const digestHex = await sha256Hex(outputRegular);
  return {
    curve,
    size,
    durationMs,
    digestHex,
    firstHex: outputRegular.length >= config.frBytes ? hex(outputRegular.slice(0, config.frBytes)) : "",
  };
}

async function runMsm(curve, size, start = 1) {
  const config = CURVE_CONFIG[curve];
  const module = await getCurveModule(curve);
  const fixture = await getFixture(curve);
  if (fixture.meta.count < size) {
    throw new Error(`fixture has ${fixture.meta.count} points, need ${size}`);
  }
  const basesPacked = fixture.bytes.slice(0, size * config.pointBytes);
  const scalarsPacked = generateRegularLEPacked(size, config.frBytes, start);
  const started = performance.now();
  const jacobianPacked = await module.msm.pippengerPackedJacobianBases(basesPacked, scalarsPacked, {
    count: 1,
    termsPerInstance: size,
    window: module.msm.bestWindow(size),
  });
  const jacobian = unpackJacobianPoint(curve, jacobianPacked.slice(0, config.pointBytes));
  const affine = await module.g1.jacobianToAffine(jacobian);
  const durationMs = performance.now() - started;
  const digestHex = await sha256Hex(concatBytes([affine.x, affine.y]));
  return {
    curve,
    size,
    durationMs,
    digestHex,
    xHex: hex(affine.x),
    yHex: hex(affine.y),
  };
}

export const curvegpuPocBridge = { init, runNtt, runMsm };

window.curvegpuPocBridge = curvegpuPocBridge;
