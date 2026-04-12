export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "./curvegpu/browser_utils.js";
import type {
  CurveGPUAffinePoint,
  CurveGPUElementBytes,
  CurveGPUJacobianPoint,
  SupportedCurveID,
} from "./index.js";
import { appendContextDiagnostics, createRequestedCurveModule, curveDisplayName, getRequestedCurveId } from "./page_library.js";

type AffinePoint = {
  x_bytes_le: string;
  y_bytes_le: string;
};

type JacobianPoint = {
  x_bytes_le: string;
  y_bytes_le: string;
  z_bytes_le: string;
};

type ScalarMulCase = {
  name: string;
  base_affine: AffinePoint;
  scalar_bytes_le: string;
  scalar_mul_affine: JacobianPoint;
};

type BaseMulCase = {
  name: string;
  scalar_bytes_le: string;
  scalar_mul_base_affine: JacobianPoint;
};

type G1ScalarMulVectors = {
  generator_affine: AffinePoint;
  scalar_cases: ScalarMulCase[];
  base_cases: BaseMulCase[];
};

type G1ScalarConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, G1ScalarConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G1 Scalar Mul Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bn254_g1_scalar_mul.json",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G1 Scalar Mul Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bls12_381_g1_scalar_mul.json?v=1",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): G1ScalarConfig {
  return CONFIGS[getRequestedCurveId()];
}

function affineFromHex(point: AffinePoint): CurveGPUAffinePoint {
  return { x: hexToBytes(point.x_bytes_le), y: hexToBytes(point.y_bytes_le) };
}

function jacobianToHex(point: CurveGPUJacobianPoint): JacobianPoint {
  return {
    x_bytes_le: bytesToHex(point.x),
    y_bytes_le: bytesToHex(point.y),
    z_bytes_le: bytesToHex(point.z),
  };
}

function expectPointBatch(name: string, got: readonly CurveGPUJacobianPoint[], want: readonly JacobianPoint[], lines: string[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = jacobianToHex(got[i]);
    if (
      gotHex.x_bytes_le !== want[i].x_bytes_le ||
      gotHex.y_bytes_le !== want[i].y_bytes_le ||
      gotHex.z_bytes_le !== want[i].z_bytes_le
    ) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
  lines.push(`${name}: OK`);
}

async function runSmoke(config: G1ScalarConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const curve = await createRequestedCurveModule(config.curve);
    const vectors = await fetchJSON<G1ScalarMulVectors>(config.vectorPath);

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push("3. Loading vectors... OK");
    lines.push(`cases.scalar = ${vectors.scalar_cases.length}`);
    lines.push(`cases.base = ${vectors.base_cases.length}`);
    lines.push("4. Initializing curve module... OK");
    writeLog(lines);

    const scalarBases = vectors.scalar_cases.map((item) => affineFromHex(item.base_affine));
    const scalarScalars = vectors.scalar_cases.map((item) => hexToBytes(item.scalar_bytes_le) as CurveGPUElementBytes);
    const scalarWant = vectors.scalar_cases.map((item) => item.scalar_mul_affine);
    expectPointBatch("scalar_mul_affine", await curve.g1.scalarMulAffineBatch(scalarBases, scalarScalars), scalarWant, lines);
    writeLog(lines);

    const generator = affineFromHex(vectors.generator_affine);
    const baseBases = Array.from({ length: vectors.base_cases.length }, () => generator);
    const baseScalars = vectors.base_cases.map((item) => hexToBytes(item.scalar_bytes_le) as CurveGPUElementBytes);
    const baseWant = vectors.base_cases.map((item) => item.scalar_mul_base_affine);
    expectPointBatch("scalar_mul_base_affine", await curve.g1.scalarMulAffineBatch(baseBases, baseScalars), baseWant, lines);
    writeLog(lines);

    lines.push("");
    lines.push(`PASS: ${curveDisplayName(config.curve)} G1 scalar mul browser smoke succeeded`);
    writeLog(lines);
    setStatus("Pass");
    setPageState("pass");
  } catch (error) {
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("Fail");
    setPageState("fail");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void runSmoke(getConfig());
});

const config = getConfig();
const params = new URLSearchParams(window.location.search);
if (params.get("autorun") === "1") {
  void runSmoke(config);
} else {
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} G1 scalar mul smoke test in browser WebGPU.`]);
}
