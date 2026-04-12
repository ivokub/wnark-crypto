export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "../../../src/curvegpu/browser_utils.js";
import type { CurveGPUElementBytes, SupportedCurveID } from "../../../src/index.js";
import { appendContextDiagnostics, createRequestedCurveModule, curveDisplayName, getRequestedCurveId } from "./shared/page_library.js";

type NTTCase = {
  name: string;
  size: number;
  input_mont_le: string[];
  forward_expected_le: string[];
  inverse_expected_le: string[];
  stage_twiddles_le: string[][];
  inverse_stage_twiddles_le: string[][];
  inverse_scale_le: string;
};

type FRNTTVectors = {
  ntt_cases: NTTCase[];
};

type NTTConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, NTTConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 fr NTT Browser Smoke",
    vectorPath: "/testdata/vectors/fr/bn254_fr_ntt.json",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 fr NTT Browser Smoke",
    vectorPath: "/testdata/vectors/fr/bls12_381_fr_ntt.json",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): NTTConfig {
  return CONFIGS[getRequestedCurveId()];
}

function bytesList(hexValues: readonly string[]): CurveGPUElementBytes[] {
  return hexValues.map(hexToBytes);
}

function expectHexBatch(name: string, got: readonly CurveGPUElementBytes[], want: readonly string[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = bytesToHex(got[i]);
    if (gotHex !== want[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${gotHex} want=${want[i]}`);
    }
  }
}

async function runSmoke(config: NTTConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const curve = await createRequestedCurveModule(config.curve);
    const vectors = await fetchJSON<FRNTTVectors>(config.vectorPath);

    lines.push("1. Requesting adapter... OK");
    appendContextDiagnostics(lines, curve.context);
    lines.push("2. Requesting device... OK");
    lines.push("3. Loading vectors... OK");
    lines.push(`cases.ntt = ${vectors.ntt_cases.length}`);
    lines.push("4. Initializing curve module... OK");
    writeLog(lines);

    for (const item of vectors.ntt_cases) {
      const input = bytesList(item.input_mont_le);
      const forward = await curve.ntt.forward(input);
      expectHexBatch(`${item.name}: forward_ntt`, forward, item.forward_expected_le);
      const inverse = await curve.ntt.inverse(forward);
      expectHexBatch(`${item.name}: inverse_ntt`, inverse, item.inverse_expected_le);
    }

    lines.push("forward_ntt: OK");
    lines.push("inverse_ntt: OK");
    lines.push("");
    lines.push(`PASS: ${curveDisplayName(config.curve)} fr NTT browser smoke succeeded`);
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
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} fr NTT smoke test in browser WebGPU.`]);
}
