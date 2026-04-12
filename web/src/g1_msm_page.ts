export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "./curvegpu/browser_utils.js";
import type {
  CurveGPUAffinePoint,
  CurveGPUElementBytes,
  CurveGPUJacobianPoint,
  CurveModule,
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

type MSMCase = {
  name: string;
  bases_affine: AffinePoint[];
  scalars_bytes_le: string[];
  expected_affine: JacobianPoint;
};

type G1MSMVectors = {
  terms_per_instance: number;
  msm_cases: MSMCase[];
  one_mont_z: string;
};

type G1MSMConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, G1MSMConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G1 MSM Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bn254_g1_msm.json",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G1 MSM Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bls12_381_g1_msm.json?v=1",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): G1MSMConfig {
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

function toAffinePoint(point: CurveGPUJacobianPoint): CurveGPUAffinePoint {
  return { x: point.x, y: point.y };
}

function expectPointBatch(name: string, got: readonly CurveGPUJacobianPoint[], want: readonly JacobianPoint[]): void {
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
      throw new Error(
        `${name}: mismatch at index ${i}` +
          ` got=(${gotHex.x_bytes_le},${gotHex.y_bytes_le},${gotHex.z_bytes_le})` +
          ` want=(${want[i].x_bytes_le},${want[i].y_bytes_le},${want[i].z_bytes_le})`,
      );
    }
  }
}

async function naiveMSMAffine(
  curve: CurveModule,
  bases: readonly CurveGPUAffinePoint[],
  scalars: readonly CurveGPUElementBytes[],
): Promise<CurveGPUJacobianPoint> {
  const scaled = await curve.g1.scalarMulAffineBatch(bases, scalars);
  if (scaled.length === 0) {
    return curve.g1.jacobianInfinity();
  }
  let accJacobian = await curve.g1.affineToJacobian(toAffinePoint(scaled[0]));
  for (let i = 1; i < scaled.length; i += 1) {
    accJacobian = await curve.g1.addMixed(accJacobian, toAffinePoint(scaled[i]));
  }
  return curve.g1.jacobianToAffine(accJacobian);
}

async function runSmoke(config: G1MSMConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const curve = await createRequestedCurveModule(config.curve);
    const vectors = await fetchJSON<G1MSMVectors>(config.vectorPath);

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push("3. Loading vectors... OK");
    lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
    lines.push(`cases.msm = ${vectors.msm_cases.length}`);
    lines.push("4. Initializing curve module... OK");
    writeLog(lines);

    const naiveResults: CurveGPUJacobianPoint[] = [];
    for (const msmCase of vectors.msm_cases) {
      naiveResults.push(
        await naiveMSMAffine(
          curve,
          msmCase.bases_affine.map(affineFromHex),
          msmCase.scalars_bytes_le.map((value) => hexToBytes(value) as CurveGPUElementBytes),
        ),
      );
    }
    expectPointBatch("msm_naive_affine", naiveResults, vectors.msm_cases.map((item) => item.expected_affine));
    lines.push("msm_naive_affine: OK");
    writeLog(lines);

    const window = 4;
    const pippengerResults = await curve.msm.pippengerAffineBatch(
      vectors.msm_cases.flatMap((item) => item.bases_affine.map(affineFromHex)),
      vectors.msm_cases.flatMap((item) => item.scalars_bytes_le.map((value) => hexToBytes(value) as CurveGPUElementBytes)),
      {
        count: vectors.msm_cases.length,
        termsPerInstance: vectors.terms_per_instance,
        window,
      },
    );
    expectPointBatch(`msm_pippenger_affine (window=${window})`, pippengerResults, vectors.msm_cases.map((item) => item.expected_affine));
    lines.push(`msm_pippenger_affine (window=${window}): OK`);
    writeLog(lines);

    lines.push("");
    lines.push(`PASS: ${curveDisplayName(config.curve)} G1 MSM browser smoke succeeded`);
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
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} G1 MSM smoke test in browser WebGPU.`]);
}
