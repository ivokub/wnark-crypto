import { createBLS12381, createBN254, createCurveGPUContext } from "/web/dist/index.js";

const CURVE_CONFIG = {
  bn254: {
    frBytes: 32,
    g1CoordinateBytes: 32,
    g1PointBytes: 96,
    g2ComponentBytes: 32,
    g2PointBytes: 192,
    g1FixtureBinPath: `/testdata/fixtures/g1/bn254_bases_jacobian.bin`,
    g1FixtureJSONPath: `/testdata/fixtures/g1/bn254_bases_jacobian.json`,
    g2FixtureBinPath: `/testdata/fixtures/g2/bn254_bases_jacobian.bin`,
    g2FixtureJSONPath: `/testdata/fixtures/g2/bn254_bases_jacobian.json`,
  },
  bls12_381: {
    frBytes: 32,
    g1CoordinateBytes: 48,
    g1PointBytes: 144,
    g2ComponentBytes: 48,
    g2PointBytes: 288,
    g1FixtureBinPath: `/testdata/fixtures/g1/bls12_381_bases_jacobian.bin`,
    g1FixtureJSONPath: `/testdata/fixtures/g1/bls12_381_bases_jacobian.json`,
    g2FixtureBinPath: `/testdata/fixtures/g2/bls12_381_bases_jacobian.bin`,
    g2FixtureJSONPath: `/testdata/fixtures/g2/bls12_381_bases_jacobian.json`,
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
        return curve === "bn254" ? createBN254(context) : createBLS12381(context);
      })(),
    );
  }
  return modulePromises.get(curve);
}

async function getFixture(curve, group) {
  const key = `${curve}:${group}`;
  if (!fixturePromises.has(key)) {
    const config = CURVE_CONFIG[curve];
    const jsonPath = group === "g1" ? config.g1FixtureJSONPath : config.g2FixtureJSONPath;
    const binPath = group === "g1" ? config.g1FixtureBinPath : config.g2FixtureBinPath;
    fixturePromises.set(
      key,
      (async () => {
        const [metaResp, binResp] = await Promise.all([
          fetch(jsonPath),
          fetch(binPath),
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
  return fixturePromises.get(key);
}

async function init(curve) {
  const started = performance.now();
  const [context, module, g1Fixture, g2Fixture] = await Promise.all([
    getContext(),
    getCurveModule(curve),
    getFixture(curve, "g1"),
    getFixture(curve, "g2"),
  ]);
  return {
    curve,
    initMs: performance.now() - started,
    adapter: {
      vendor: context.diagnostics.vendor ?? "",
      architecture: context.diagnostics.architecture ?? "",
      description: context.diagnostics.description ?? "",
    },
    g1FixturePoints: g1Fixture.meta.count,
    g2FixturePoints: g2Fixture.meta.count,
    g1Window: module.msm.bestWindow(1 << 12),
    g2Window: module.g2msm.bestWindow(1 << 12),
  };
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

async function runG1Msm(curve, size, start = 1) {
  const config = CURVE_CONFIG[curve];
  const module = await getCurveModule(curve);
  const fixture = await getFixture(curve, "g1");
  if (fixture.meta.count < size) {
    throw new Error(`fixture has ${fixture.meta.count} points, need ${size}`);
  }
  const basesPacked = fixture.bytes.slice(0, size * config.g1PointBytes);
  const scalarsPacked = generateRegularLEPacked(size, config.frBytes, start);
  const started = performance.now();
  const jacobianPacked = await module.msm.pippengerPackedJacobianBases(basesPacked, scalarsPacked, {
    count: 1,
    termsPerInstance: size,
    window: module.msm.bestWindow(size),
  });
  const jacobian = unpackG1JacobianPoint(curve, jacobianPacked.slice(0, config.g1PointBytes));
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

async function runG2Msm(curve, size, start = 1) {
  const config = CURVE_CONFIG[curve];
  const module = await getCurveModule(curve);
  const fixture = await getFixture(curve, "g2");
  if (fixture.meta.count < size) {
    throw new Error(`g2 fixture has ${fixture.meta.count} points, need ${size}`);
  }
  const basesPacked = fixture.bytes.slice(0, size * config.g2PointBytes);
  const scalarsPacked = generateRegularLEPacked(size, config.frBytes, start);
  const started = performance.now();
  const jacobianPacked = await module.g2msm.pippengerPackedJacobianBases(basesPacked, scalarsPacked, {
    count: 1,
    termsPerInstance: size,
    window: module.g2msm.bestWindow(size),
  });
  const jacobian = unpackG2JacobianPoint(curve, jacobianPacked.slice(0, config.g2PointBytes));
  const affine = await module.g2.jacobianToAffine(jacobian);
  const durationMs = performance.now() - started;
  const digestHex = await sha256Hex(concatBytes([affine.x.c0, affine.x.c1, affine.y.c0, affine.y.c1]));
  return {
    curve,
    size,
    durationMs,
    digestHex,
    xC0Hex: hex(affine.x.c0),
    xC1Hex: hex(affine.x.c1),
    yC0Hex: hex(affine.y.c0),
    yC1Hex: hex(affine.y.c1),
  };
}

async function prewarm(curve, nttSize, g1MsmSize, g2MsmSize) {
  const started = performance.now();
  const ntt = await runNtt(curve, nttSize, 1);
  const g1 = await runG1Msm(curve, g1MsmSize, 1);
  const g2 = await runG2Msm(curve, g2MsmSize, 1);
  return {
    curve,
    totalMs: performance.now() - started,
    nttMs: ntt.durationMs,
    g1MsmMs: g1.durationMs,
    g2MsmMs: g2.durationMs,
  };
}

export const curvegpuPocBridge = { init, prewarm, runNtt, runG1Msm, runG2Msm };

window.curvegpuPocBridge = curvegpuPocBridge;
