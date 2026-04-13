export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "../../../src/curvegpu/browser_utils.js";
import type {
  CurveGPUElementBytes,
  CurveGPUFp2Element,
  CurveGPUG2AffinePoint,
  CurveGPUG2JacobianPoint,
  CurveModule,
  SupportedCurveID,
} from "../../../src/index.js";
import { appendContextDiagnostics, createRequestedCurveModule, curveDisplayName, getRequestedCurveId } from "./shared/page_library.js";

type Fp2Point = {
  c0_bytes_le: string;
  c1_bytes_le: string;
};

type AffinePoint = {
  x: Fp2Point;
  y: Fp2Point;
};

type JacobianPoint = {
  x: Fp2Point;
  y: Fp2Point;
  z: Fp2Point;
};

type MSMCase = {
  name: string;
  bases_affine: AffinePoint[];
  scalars_bytes_le: string[];
  expected_affine: JacobianPoint;
};

type G2MSMVectors = {
  terms_per_instance: number;
  msm_cases: MSMCase[];
};

type G2MSMConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, G2MSMConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G2 MSM Browser Smoke",
    vectorPath: "/testdata/vectors/g2/bn254_g2_msm.json?v=1",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G2 MSM Browser Smoke",
    vectorPath: "/testdata/vectors/g2/bls12_381_g2_msm.json?v=1",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): G2MSMConfig {
  return CONFIGS[getRequestedCurveId()];
}

function fp2FromHex(point: Fp2Point): CurveGPUFp2Element {
  return { c0: hexToBytes(point.c0_bytes_le), c1: hexToBytes(point.c1_bytes_le) };
}

function affineFromHex(point: AffinePoint): CurveGPUG2AffinePoint {
  return { x: fp2FromHex(point.x), y: fp2FromHex(point.y) };
}

function affineToHex(point: CurveGPUG2AffinePoint): AffinePoint {
  return {
    x: { c0_bytes_le: bytesToHex(point.x.c0), c1_bytes_le: bytesToHex(point.x.c1) },
    y: { c0_bytes_le: bytesToHex(point.y.c0), c1_bytes_le: bytesToHex(point.y.c1) },
  };
}

function jacobianToHex(point: CurveGPUG2JacobianPoint): JacobianPoint {
  return {
    x: { c0_bytes_le: bytesToHex(point.x.c0), c1_bytes_le: bytesToHex(point.x.c1) },
    y: { c0_bytes_le: bytesToHex(point.y.c0), c1_bytes_le: bytesToHex(point.y.c1) },
    z: { c0_bytes_le: bytesToHex(point.z.c0), c1_bytes_le: bytesToHex(point.z.c1) },
  };
}

function toAffinePoint(point: CurveGPUG2JacobianPoint): CurveGPUG2AffinePoint {
  return { x: point.x, y: point.y };
}

function equalFp2(a: Fp2Point, b: Fp2Point): boolean {
  return a.c0_bytes_le === b.c0_bytes_le && a.c1_bytes_le === b.c1_bytes_le;
}

function expectPointBatch(name: string, got: readonly CurveGPUG2JacobianPoint[], want: readonly JacobianPoint[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = jacobianToHex(got[i]);
    if (!equalFp2(gotHex.x, want[i].x) || !equalFp2(gotHex.y, want[i].y) || !equalFp2(gotHex.z, want[i].z)) {
      throw new Error(
        `${name}: mismatch at index ${i}` +
          ` got=(${gotHex.x.c0_bytes_le}/${gotHex.x.c1_bytes_le},${gotHex.y.c0_bytes_le}/${gotHex.y.c1_bytes_le},${gotHex.z.c0_bytes_le}/${gotHex.z.c1_bytes_le})` +
          ` want=(${want[i].x.c0_bytes_le}/${want[i].x.c1_bytes_le},${want[i].y.c0_bytes_le}/${want[i].y.c1_bytes_le},${want[i].z.c0_bytes_le}/${want[i].z.c1_bytes_le})`,
      );
    }
  }
}

function expectAffineBatch(name: string, got: readonly CurveGPUG2AffinePoint[], want: readonly JacobianPoint[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = affineToHex(got[i]);
    if (!equalFp2(gotHex.x, want[i].x) || !equalFp2(gotHex.y, want[i].y)) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
}

async function naiveMSMAffine(
  curve: CurveModule,
  bases: readonly CurveGPUG2AffinePoint[],
  scalars: readonly CurveGPUElementBytes[],
): Promise<CurveGPUG2AffinePoint> {
  const scaled = await curve.g2.scalarMulAffineBatch(bases, scalars);
  if (scaled.length === 0) {
    return curve.g2.affineInfinity();
  }
  let accJacobian = await curve.g2.affineToJacobian(toAffinePoint(scaled[0]));
  for (let i = 1; i < scaled.length; i += 1) {
    accJacobian = await curve.g2.addMixed(accJacobian, toAffinePoint(scaled[i]));
  }
  return curve.g2.jacobianToAffine(accJacobian);
}

async function runSmoke(config: G2MSMConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const curve = await createRequestedCurveModule(config.curve);
    const vectors = await fetchJSON<G2MSMVectors>(config.vectorPath);

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push("3. Loading vectors... OK");
    lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
    lines.push(`cases.msm = ${vectors.msm_cases.length}`);
    lines.push("4. Initializing curve module... OK");
    writeLog(lines);

    const naiveResults: CurveGPUG2AffinePoint[] = [];
    for (const msmCase of vectors.msm_cases) {
      naiveResults.push(
        await naiveMSMAffine(
          curve,
          msmCase.bases_affine.map(affineFromHex),
          msmCase.scalars_bytes_le.map((value) => hexToBytes(value) as CurveGPUElementBytes),
        ),
      );
    }
    expectAffineBatch("msm_naive_affine", naiveResults, vectors.msm_cases.map((item) => item.expected_affine));
    lines.push("msm_naive_affine: OK");
    writeLog(lines);

    const window = 4;
    const pippengerResults = await curve.g2msm.pippengerAffineBatch(
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
    lines.push(`PASS: ${curveDisplayName(config.curve)} G2 MSM browser smoke succeeded`);
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
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} G2 MSM smoke test in browser WebGPU.`]);
}
