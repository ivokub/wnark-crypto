export {};

import { bytesToHex, createPageUI, fetchJSON, hexToBytes } from "./curvegpu/browser_utils.js";
import type { CurveGPUElementBytes, FieldModule, SupportedCurveID } from "./index.js";
import { appendContextDiagnostics, createRequestedCurveModule, curveDisplayName, getRequestedCurveId } from "./page_library.js";

type ElementCase = {
  name: string;
  a_bytes_le: string;
  b_bytes_le: string;
  equal_bytes_le: string;
  add_bytes_le: string;
  sub_bytes_le: string;
  neg_a_bytes_le: string;
  double_a_bytes_le: string;
  mul_bytes_le: string;
  square_a_bytes_le: string;
};

type NormalizeCase = {
  name: string;
  input_bytes_le: string;
  expected_bytes_le: string;
};

type ConvertCase = {
  name: string;
  regular_bytes_le: string;
  mont_bytes_le: string;
};

type FPOpsVectors = {
  element_cases: ElementCase[];
  edge_cases: ElementCase[];
  differential_cases: ElementCase[];
  normalize_cases: NormalizeCase[];
  convert_cases: ConvertCase[];
};

type FpOpsConfig = {
  curve: SupportedCurveID;
  title: string;
  vectorPath: string;
};

const CONFIGS: Record<SupportedCurveID, FpOpsConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 fp Ops Browser Smoke",
    vectorPath: "/testdata/vectors/fp/bn254_fp_ops.json",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 fp Ops Browser Smoke",
    vectorPath: "/testdata/vectors/fp/bls12_381_fp_ops.json?v=1",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): FpOpsConfig {
  return CONFIGS[getRequestedCurveId()];
}

function combineElementCases(vectors: FPOpsVectors): ElementCase[] {
  return [...vectors.element_cases, ...vectors.edge_cases, ...vectors.differential_cases];
}

function zeroHex(byteSize: number): string {
  return bytesToHex(new Uint8Array(byteSize));
}

function isNonZeroHex(hex: string): boolean {
  return hexToBytes(hex).some((byte) => byte !== 0);
}

function bytesList(hexValues: readonly string[]): Uint8Array[] {
  return hexValues.map(hexToBytes);
}

function expectHexBatch(name: string, got: readonly CurveGPUElementBytes[], wantHex: readonly string[], lines: string[]): void {
  if (got.length !== wantHex.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${wantHex.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    const gotHex = bytesToHex(got[i]);
    if (gotHex !== wantHex[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${gotHex} want=${wantHex[i]}`);
    }
  }
  lines.push(`${name}: OK`);
}

function expectBoolBatch(name: string, got: readonly boolean[], want: readonly boolean[], lines: string[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i] !== want[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${String(got[i])} want=${String(want[i])}`);
    }
  }
  lines.push(`${name}: OK`);
}

function mustFindConvertCase(cases: readonly ConvertCase[], name: string): ConvertCase {
  const found = cases.find((item) => item.name === name);
  if (!found) {
    throw new Error(`missing convert case ${name}`);
  }
  return found;
}

async function runBrowserSmoke(config: FpOpsConfig): Promise<string[]> {
  const lines: string[] = [`=== ${config.title} ===`, ""];
  writeLog(lines);

  const curve = await createRequestedCurveModule(config.curve);
  const vectors = await fetchJSON<FPOpsVectors>(config.vectorPath);

  lines.push("1. Requesting adapter... OK");
  appendContextDiagnostics(lines, curve.context);
  lines.push("2. Requesting device... OK");
  lines.push("3. Loading vectors... OK");
  lines.push(`cases.sanity = ${vectors.element_cases.length}`);
  lines.push(`cases.edge = ${vectors.edge_cases.length}`);
  lines.push(`cases.differential = ${vectors.differential_cases.length}`);
  lines.push(`cases.normalize = ${vectors.normalize_cases.length}`);
  lines.push(`cases.convert = ${vectors.convert_cases.length}`);
  writeLog(lines);

  const fp: FieldModule = curve.fp;
  const elementCases = combineElementCases(vectors);
  const aHex = elementCases.map((item) => item.a_bytes_le);
  const bHex = elementCases.map((item) => item.b_bytes_le);
  const aBytes = bytesList(aHex);
  const bBytes = bytesList(bHex);
  const zeroHexValue = zeroHex(fp.byteSize);
  const oneMontHex = bytesToHex(await fp.montOne());

  expectHexBatch("copy", await fp.copyBatch(aBytes), aHex, lines);
  writeLog(lines);
  expectBoolBatch("equal", await fp.equalBatch(aBytes, bBytes), elementCases.map((item) => isNonZeroHex(item.equal_bytes_le)), lines);
  writeLog(lines);
  expectHexBatch("zero", Array.from({ length: elementCases.length }, () => fp.zero()), Array.from({ length: elementCases.length }, () => zeroHexValue), lines);
  writeLog(lines);
  const oneBatch = Array.from({ length: elementCases.length }, () => hexToBytes(oneMontHex));
  expectHexBatch("one", oneBatch, Array.from({ length: elementCases.length }, () => oneMontHex), lines);
  writeLog(lines);
  expectHexBatch("add", await fp.addBatch(aBytes, bBytes), elementCases.map((item) => item.add_bytes_le), lines);
  writeLog(lines);
  expectHexBatch("sub", await fp.subBatch(aBytes, bBytes), elementCases.map((item) => item.sub_bytes_le), lines);
  writeLog(lines);
  expectHexBatch("neg", await fp.negBatch(aBytes), elementCases.map((item) => item.neg_a_bytes_le), lines);
  writeLog(lines);
  expectHexBatch("double", await fp.doubleBatch(aBytes), elementCases.map((item) => item.double_a_bytes_le), lines);
  writeLog(lines);
  expectHexBatch("mul", await fp.mulBatch(aBytes, bBytes), elementCases.map((item) => item.mul_bytes_le), lines);
  writeLog(lines);
  expectHexBatch("square", await fp.squareBatch(aBytes), elementCases.map((item) => item.square_a_bytes_le), lines);
  writeLog(lines);
  expectHexBatch(
    "to_mont",
    await fp.toMontgomeryBatch(bytesList(vectors.convert_cases.map((item) => item.regular_bytes_le))),
    vectors.convert_cases.map((item) => item.mont_bytes_le),
    lines,
  );
  writeLog(lines);
  expectHexBatch(
    "from_mont",
    await fp.fromMontgomeryBatch(bytesList(vectors.convert_cases.map((item) => item.mont_bytes_le))),
    vectors.convert_cases.map((item) => item.regular_bytes_le),
    lines,
  );
  writeLog(lines);
  expectHexBatch(
    "normalize",
    await fp.normalizeMontBatch(bytesList(vectors.normalize_cases.map((item) => item.input_bytes_le))),
    vectors.normalize_cases.map((item) => item.expected_bytes_le),
    lines,
  );
  writeLog(lines);

  const oneCase = mustFindConvertCase(vectors.convert_cases, "one");
  const oneHexExpected = bytesToHex(await fp.montOne());
  if (oneHexExpected !== oneCase.mont_bytes_le) {
    throw new Error(`one: mismatch got=${oneHexExpected} want=${oneCase.mont_bytes_le}`);
  }

  lines.push("");
  lines.push(`PASS: ${curveDisplayName(config.curve)} fp browser smoke succeeded`);
  writeLog(lines);
  return lines;
}

async function main(): Promise<void> {
  const config = getConfig();
  runButton.disabled = true;
  setPageState("running");
  setStatus("Running");

  try {
    const lines = await runBrowserSmoke(config);
    writeLog(lines);
    setPageState("pass");
    setStatus("Pass");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeLog([`=== ${config.title} ===`, "", `FAIL: ${message}`]);
    setPageState("fail");
    setStatus("Fail");
    throw error;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void main();
});

const config = getConfig();
const params = new URLSearchParams(window.location.search);
if (params.get("autorun") === "1") {
  void main();
} else {
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} fp smoke test in browser WebGPU.`]);
}
