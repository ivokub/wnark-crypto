export {};

import {
  bytesToHex,
  createPageUI,
  fetchJSON,
  hexToBytes,
  mustElement,
  yieldToBrowser,
} from "../../../src/curvegpu/browser_utils.js";
import { benchmarkTotalDuration } from "./shared/bench_total.js";
import { createPreferredByteBaseSource } from "../../../src/curvegpu/msm_bench_sources.js";
import { createOptimizedG2MSMBenchModule } from "../../../src/curvegpu/g2_msm_bench_optimized.js";
import { createJacG2MSMModule } from "../../../src/curvegpu/g2_msm_jac.js";
import { makeRandomScalarBatch } from "../../../src/curvegpu/msm_shared.js";
import type {
  CurveGPUElementBytes,
  CurveGPUFp2Element,
  CurveGPUG2AffinePoint,
  CurveGPUG2JacobianPoint,
  SupportedCurveID,
} from "../../../src/index.js";
import { appendContextDiagnostics, createRequestedCurveModule } from "./shared/page_library.js";

type Fp2Point = {
  c0_bytes_le: string;
  c1_bytes_le: string;
};

type AffinePoint = {
  x: Fp2Point;
  y: Fp2Point;
};

type G2Case = {
  name: string;
  p_affine: AffinePoint;
  q_affine: AffinePoint;
};

type G2OpsVectors = {
  point_cases: G2Case[];
};

type CurveBenchConfig = {
  curve: SupportedCurveID;
  title: string;
  successMessage: string;
  componentBytes: number;
  pointBytes: number;
  opsVectorsPath: string;
  fixtureJSONPath?: string;
  fixtureBinPath?: string;
};

const CURVE_CONFIGS: Record<SupportedCurveID, CurveBenchConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G2 MSM Browser Benchmark",
    successMessage: "BN254 G2 MSM browser benchmark completed",
    componentBytes: 32,
    pointBytes: 192,
    opsVectorsPath: "/testdata/vectors/g2/bn254_g2_ops.json?v=1",
    fixtureJSONPath: "/testdata/fixtures/g2/bn254_bases_jacobian.json?v=1",
    fixtureBinPath: "/testdata/fixtures/g2/bn254_bases_jacobian.bin?v=1",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G2 MSM Browser Benchmark",
    successMessage: "BLS12-381 G2 MSM browser benchmark completed",
    componentBytes: 48,
    pointBytes: 288,
    opsVectorsPath: "/testdata/vectors/g2/bls12_381_g2_ops.json?v=1",
    fixtureJSONPath: "/testdata/fixtures/g2/bls12_381_bases_jacobian.json?v=1",
    fixtureBinPath: "/testdata/fixtures/g2/bls12_381_bases_jacobian.bin?v=1",
  },
};

const minLogEl = document.getElementById("min-log") as HTMLInputElement | null;
const maxLogEl = document.getElementById("max-log") as HTMLInputElement | null;
const itersEl = document.getElementById("iters") as HTMLInputElement | null;
const runButton = document.getElementById("run") as HTMLButtonElement | null;
const statusEl = document.getElementById("status") as HTMLElement | null;
const logEl = document.getElementById("log") as HTMLElement | null;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(curve: SupportedCurveID): CurveBenchConfig {
  return CURVE_CONFIGS[curve];
}

function fp2FromHex(point: Fp2Point): CurveGPUFp2Element {
  return { c0: hexToBytes(point.c0_bytes_le), c1: hexToBytes(point.c1_bytes_le) };
}

function affineFromHex(point: AffinePoint): CurveGPUG2AffinePoint {
  return { x: fp2FromHex(point.x), y: fp2FromHex(point.y) };
}

function isAffineInfinity(point: CurveGPUG2AffinePoint): boolean {
  return (
    point.x.c0.every((byte) => byte === 0) &&
    point.x.c1.every((byte) => byte === 0) &&
    point.y.c0.every((byte) => byte === 0) &&
    point.y.c1.every((byte) => byte === 0)
  );
}

function findGeneratorPoint(vectors: G2OpsVectors): CurveGPUG2AffinePoint {
  for (const item of vectors.point_cases) {
    for (const point of [item.p_affine, item.q_affine]) {
      const parsed = affineFromHex(point);
      if (!isAffineInfinity(parsed)) {
        return parsed;
      }
    }
  }
  throw new Error("no non-infinity G2 point found in vectors");
}

function makeScalarHexLEFromUint64(value: bigint): string {
  const out = new Uint8Array(32);
  let x = value;
  for (let i = 0; i < 8; i += 1) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return bytesToHex(out);
}

function packJacobianPoints(
  points: readonly CurveGPUG2JacobianPoint[],
  componentBytes: number,
  pointBytes: number,
): Uint8Array {
  const out = new Uint8Array(points.length * pointBytes);
  points.forEach((point, index) => {
    const base = index * pointBytes;
    out.set(point.x.c0, base);
    out.set(point.x.c1, base + componentBytes);
    out.set(point.y.c0, base + 2 * componentBytes);
    out.set(point.y.c1, base + 3 * componentBytes);
    out.set(point.z.c0, base + 4 * componentBytes);
    out.set(point.z.c1, base + 5 * componentBytes);
  });
  return out;
}

function makeMSMScalarsPacked(count: number): Uint8Array {
  const out = new Uint8Array(count * 32);
  const hexes = makeRandomScalarBatch(count).hexes;
  for (let i = 0; i < count; i += 1) {
    out.set(hexToBytes(hexes[i]), i * 32);
  }
  return out;
}

function fixtureGenerationHint(curve: SupportedCurveID, size: number): string {
  return `make fixture-${curve}-g2 G2_COUNT=${size}`;
}

async function buildGeneratedBases(
  curve: Awaited<ReturnType<typeof createRequestedCurveModule>>,
  config: CurveBenchConfig,
  generator: CurveGPUG2AffinePoint,
  count: number,
): Promise<Uint8Array> {
  const bases = Array.from({ length: count }, () => generator);
  const scalars = Array.from(
    { length: count },
    (_, index) => hexToBytes(makeScalarHexLEFromUint64(BigInt(index + 1))) as CurveGPUElementBytes,
  );
  const generated = await curve.g2.scalarMulAffineBatch(bases, scalars);
  return packJacobianPoints(generated, config.componentBytes, config.pointBytes);
}

async function runBenchmark(): Promise<void> {
  const params = new URLSearchParams(window.location.search);
  const curveId = (params.get("curve") ?? "bn254") as SupportedCurveID;
  const config = getConfig(curveId);
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  mustElement(runButton, "run").disabled = true;

  try {
    const minLog = Number.parseInt(mustElement(minLogEl, "min-log").value, 10);
    const maxLog = Number.parseInt(mustElement(maxLogEl, "max-log").value, 10);
    const iters = Number.parseInt(mustElement(itersEl, "iters").value, 10);
    if (!Number.isInteger(minLog) || !Number.isInteger(maxLog) || !Number.isInteger(iters) || minLog < 1 || maxLog < minLog || iters < 1) {
      throw new Error("invalid benchmark controls");
    }

    const initStart = performance.now();
    const curve = await createRequestedCurveModule(config.curve);
    const optimizedMSM = createOptimizedG2MSMBenchModule(curve.context, config.curve, config.pointBytes);
    const jacMSM = createJacG2MSMModule(curve.context, config.curve, config.pointBytes);
    const g2Vectors = await fetchJSON<G2OpsVectors>(config.opsVectorsPath);
    const generator = findGeneratorPoint(g2Vectors);
    const baseSourceProvider = createPreferredByteBaseSource({
      locationSearch: window.location.search,
      pointBytes: config.pointBytes,
      fixtureJSONPath: config.fixtureJSONPath,
      fixtureBinPath: config.fixtureBinPath,
      generatedLoadBases: async (size) => buildGeneratedBases(curve, config, generator, size),
      generateHint: (size) => fixtureGenerationHint(config.curve, size <= 0 ? (1 << 14) : size),
      fixtureLabel: "G2 base",
    });
    const baseSourceInit = await baseSourceProvider.init();
    const initMs = performance.now() - initStart;

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push(`3. Loading base source... OK (${baseSourceInit.context.baseSource})`);
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    if (baseSourceInit.postMetricLines) {
      lines.push(...baseSourceInit.postMetricLines);
    }
    const prewarmSize = 1 << minLog;
    lines.push(`4. Prewarming optimized G2 MSM runtime at size ${prewarmSize}...`);
    writeLog(lines);
    await yieldToBrowser();
    {
      const { bases: prewarmBases } = await baseSourceProvider.loadBases({
        context: baseSourceInit.context,
        size: prewarmSize,
      });
      const prewarmScalars = makeMSMScalarsPacked(prewarmSize);
      const prewarmWindow = optimizedMSM.bestWindow(prewarmSize);
      await optimizedMSM.pippengerPackedJacobianBases(prewarmBases, prewarmScalars, {
        count: 1,
        termsPerInstance: prewarmSize,
        window: prewarmWindow,
      });
    }
    lines[lines.length - 1] = `4. Prewarming optimized G2 MSM runtime at size ${prewarmSize}... OK`;
    lines.push(`5. Prewarming Jacobian G2 MSM runtime at size ${prewarmSize}...`);
    writeLog(lines);
    await yieldToBrowser();
    {
      const { bases: prewarmBases } = await baseSourceProvider.loadBases({
        context: baseSourceInit.context,
        size: prewarmSize,
      });
      const prewarmScalars = makeMSMScalarsPacked(prewarmSize);
      const prewarmWindow = jacMSM.bestWindow(prewarmSize);
      await jacMSM.pippengerPackedJacobianBases(prewarmBases, prewarmScalars, {
        count: 1,
        termsPerInstance: prewarmSize,
        window: prewarmWindow,
      });
    }
    lines[lines.length - 1] = `5. Prewarming Jacobian G2 MSM runtime at size ${prewarmSize}... OK`;
    lines.push("");
    lines.push("size,op,window,init_ms,prep_ms,cold_total_ms,cold_with_init_prep_ms,warm_total_ms");
    writeLog(lines);
    await yieldToBrowser();

    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      await yieldToBrowser();
      const size = 1 << logSize;
      const prepStart = performance.now();
      const { bases: baseBytes } = await baseSourceProvider.loadBases({
        context: baseSourceInit.context,
        size,
      });
      const scalarsPacked = makeMSMScalarsPacked(size);
      const prepMs = performance.now() - prepStart;
      const window = optimizedMSM.bestWindow(size);
      const benchmark = await benchmarkTotalDuration(iters, async () => {
        await optimizedMSM.pippengerPackedJacobianBases(baseBytes, scalarsPacked, {
          count: 1,
          termsPerInstance: size,
          window,
        });
      }, yieldToBrowser);
      lines.push(
        [
          `${size}`,
          "msm_pippenger_packed",
          `${window}`,
          initMs.toFixed(3),
          prepMs.toFixed(3),
          benchmark.coldMs.toFixed(3),
          (initMs + prepMs + benchmark.coldMs).toFixed(3),
          benchmark.warmMs.toFixed(3),
        ].join(","),
      );
      writeLog(lines);

      const jacBenchmark = await benchmarkTotalDuration(iters, async () => {
        await jacMSM.pippengerPackedJacobianBases(baseBytes, scalarsPacked, {
          count: 1,
          termsPerInstance: size,
          window,
        });
      }, yieldToBrowser);
      lines.push(
        [
          `${size}`,
          "msm_jac_pippenger_packed",
          `${window}`,
          initMs.toFixed(3),
          prepMs.toFixed(3),
          jacBenchmark.coldMs.toFixed(3),
          (initMs + prepMs + jacBenchmark.coldMs).toFixed(3),
          jacBenchmark.warmMs.toFixed(3),
        ].join(","),
      );
      writeLog(lines);
    }

    lines.push("");
    lines.push(`PASS: ${config.successMessage}`);
    writeLog(lines);
    setStatus("Pass");
    setPageState("pass");
  } catch (error) {
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("Fail");
    setPageState("fail");
  } finally {
    mustElement(runButton, "run").disabled = false;
  }
}

mustElement(runButton, "run").addEventListener("click", () => {
  void runBenchmark();
});

const params = new URLSearchParams(window.location.search);
const curveId = (params.get("curve") ?? "bn254") as SupportedCurveID;
const config = getConfig(curveId);
if (params.get("autorun") === "1") {
  void runBenchmark();
} else {
  writeLog([`=== ${config.title} ===`, "", `Press Run to benchmark ${config.curve} G2 MSM in browser WebGPU.`]);
}
