export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "../../../src/curvegpu/browser_utils.js";
import type {
  CurveGPUAffinePoint,
  CurveGPUJacobianPoint,
  G1Module,
  SupportedCurveID,
} from "../../../src/index.js";
import { appendContextDiagnostics, createRequestedCurveModule, curveDisplayName, getRequestedCurveId } from "./shared/page_library.js";

type AffinePoint = {
  x_bytes_le: string;
  y_bytes_le: string;
};

type JacobianPoint = {
  x_bytes_le: string;
  y_bytes_le: string;
  z_bytes_le: string;
};

type G1Case = {
  name: string;
  p_affine: AffinePoint;
  q_affine: AffinePoint;
  p_jacobian: JacobianPoint;
  p_affine_output: JacobianPoint;
  neg_p_jacobian: JacobianPoint;
  double_p_jacobian: JacobianPoint;
  add_mixed_p_plus_q_jacobian: JacobianPoint;
  affine_add_p_plus_q: JacobianPoint;
};

type G1OpsVectors = {
  point_cases: G1Case[];
};

type G1OpsConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, G1OpsConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G1 Ops Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bn254_g1_ops.json",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G1 Ops Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bls12_381_g1_ops.json?v=1",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): G1OpsConfig {
  return CONFIGS[getRequestedCurveId()];
}

function affineFromHex(point: AffinePoint): CurveGPUAffinePoint {
  return { x: hexToBytes(point.x_bytes_le), y: hexToBytes(point.y_bytes_le) };
}

function jacobianFromHex(point: JacobianPoint): CurveGPUJacobianPoint {
  return {
    x: hexToBytes(point.x_bytes_le),
    y: hexToBytes(point.y_bytes_le),
    z: hexToBytes(point.z_bytes_le),
  };
}

function jacobianToHex(point: CurveGPUJacobianPoint): JacobianPoint {
  return {
    x_bytes_le: bytesToHex(point.x),
    y_bytes_le: bytesToHex(point.y),
    z_bytes_le: bytesToHex(point.z),
  };
}

function affineToHex(point: CurveGPUAffinePoint): AffinePoint {
  return {
    x_bytes_le: bytesToHex(point.x),
    y_bytes_le: bytesToHex(point.y),
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

function expectAffineBatch(name: string, got: readonly CurveGPUAffinePoint[], want: readonly AffinePoint[], lines: string[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = affineToHex(got[i]);
    if (gotHex.x_bytes_le !== want[i].x_bytes_le || gotHex.y_bytes_le !== want[i].y_bytes_le) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
  lines.push(`${name}: OK`);
}

async function runSmoke(config: G1OpsConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const curve = await createRequestedCurveModule(config.curve);
    const vectors = await fetchJSON<G1OpsVectors>(config.vectorPath);
    const g1: G1Module = curve.g1;

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push("3. Loading vectors... OK");
    lines.push(`cases.g1 = ${vectors.point_cases.length}`);
    lines.push("4. Initializing curve module... OK");
    writeLog(lines);

    const pAffine = vectors.point_cases.map((item) => affineFromHex(item.p_affine));
    const qAffine = vectors.point_cases.map((item) => affineFromHex(item.q_affine));
    const pJacobian = vectors.point_cases.map((item) => jacobianFromHex(item.p_jacobian));
    const negWant = vectors.point_cases.map((item) => item.neg_p_jacobian);
    const doubleWant = vectors.point_cases.map((item) => item.double_p_jacobian);
    const addWant = vectors.point_cases.map((item) => item.add_mixed_p_plus_q_jacobian);
    const affineWant = vectors.point_cases.map((item) => ({
      x_bytes_le: item.p_affine_output.x_bytes_le,
      y_bytes_le: item.p_affine_output.y_bytes_le,
    }));
    const affineAddWant = vectors.point_cases.map((item) => item.affine_add_p_plus_q);
    const oneMont = await curve.fp.montOne();
    const jacInfinityWant = vectors.point_cases.map(() => ({
      x_bytes_le: bytesToHex(oneMont),
      y_bytes_le: bytesToHex(oneMont),
      z_bytes_le: bytesToHex(curve.fp.zero()),
    }));

    expectPointBatch("copy", await g1.copyBatch(pJacobian), vectors.point_cases.map((item) => item.p_jacobian), lines);
    writeLog(lines);
    expectPointBatch("jac_infinity", await g1.jacobianInfinityBatch(vectors.point_cases.length), jacInfinityWant, lines);
    writeLog(lines);
    expectPointBatch("affine_to_jac", await g1.affineToJacobianBatch(pAffine), vectors.point_cases.map((item) => item.p_jacobian), lines);
    writeLog(lines);
    expectPointBatch("neg_jac", await g1.negJacobianBatch(pJacobian), negWant, lines);
    writeLog(lines);
    expectAffineBatch("jac_to_affine", await g1.jacobianToAffineBatch(pJacobian), affineWant, lines);
    writeLog(lines);
    expectPointBatch("double_jac", await g1.doubleJacobianBatch(pJacobian), doubleWant, lines);
    writeLog(lines);
    expectPointBatch("add_mixed", await g1.addMixedBatch(pJacobian, qAffine), addWant, lines);
    writeLog(lines);
    expectPointBatch("affine_add", await g1.affineAddBatch(pAffine, qAffine), affineAddWant, lines);
    writeLog(lines);

    lines.push("");
    lines.push(`PASS: ${curveDisplayName(config.curve)} G1 browser smoke succeeded`);
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
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} G1 smoke test in browser WebGPU.`]);
}
