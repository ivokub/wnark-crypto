export {};

import {
  createCurveGPUContext,
  createCurveModule,
  type CurveGPUAffinePoint,
  type CurveGPUElementBytes,
  type SupportedCurveID,
} from "../../src/index.js";

const EXAMPLE_BASES: Record<SupportedCurveID, CurveGPUAffinePoint> = {
  bn254: {
    x: hexToBytes("9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e"),
    y: hexToBytes("3a1b1e8b1b87baa67b168eeb51d6f114588cf2f0de46ddcc5ebe0f3483ef141c"),
  },
  bls12_381: {
    x: hexToBytes("160c53fd9087b35cf5ff769967fc1778c1a13b14c7954f1547e7d0f3cd6aaef040f4db21cc6eceed75fb0b9e41770112"),
    y: hexToBytes("7122e70cd593acba8efd18791a63228cce250757135f59dd945140502958ac51c05900ad3f8c1c0e6aa20850fc3ebc0b"),
  },
};

function mustElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLElement)) {
    throw new Error(`missing element: ${id}`);
  }
  return el as T;
}

function getCurveId(): SupportedCurveID {
  const curve = new URLSearchParams(window.location.search).get("curve") ?? "bn254";
  if (curve !== "bn254" && curve !== "bls12_381") {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return curve;
}

function curveLabel(curve: SupportedCurveID): string {
  return curve === "bn254" ? "BN254" : "BLS12-381";
}

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new Error(`invalid hex length ${hex.length}`);
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function shortHex(bytes: Uint8Array, keep = 12): string {
  const hex = bytesToHex(bytes);
  return hex.length <= keep * 2 ? hex : `${hex.slice(0, keep)}...${hex.slice(-keep)}`;
}

function scalarLE(value: number): CurveGPUElementBytes {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`invalid scalar: ${value}`);
  }
  const out = new Uint8Array(32);
  let carry = value;
  let i = 0;
  while (carry > 0 && i < out.length) {
    out[i] = carry & 0xff;
    carry >>>= 8;
    i += 1;
  }
  return out;
}

function equalElements(a: readonly Uint8Array[], b: readonly Uint8Array[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (bytesToHex(a[i]) !== bytesToHex(b[i])) {
      return false;
    }
  }
  return true;
}

function equalAffine(a: CurveGPUAffinePoint, b: CurveGPUAffinePoint): boolean {
  return bytesToHex(a.x) === bytesToHex(b.x) && bytesToHex(a.y) === bytesToHex(b.y);
}

async function buildNTTInput(one: Uint8Array, add: (a: Uint8Array, b: Uint8Array) => Promise<Uint8Array>, zero: Uint8Array): Promise<Uint8Array[]> {
  const two = await add(one, one);
  const three = await add(two, one);
  const four = await add(two, two);
  return [one, two, three, four, zero, zero, zero, zero];
}

async function runExample(curveId: SupportedCurveID, log: (lines: string[]) => void): Promise<void> {
  const lines = [`=== ${curveLabel(curveId)} CurveGPU Library Example ===`, ""];
  log(lines);

  const context = await createCurveGPUContext();
  const curve = createCurveModule(context, curveId);
  const diagnostics = context.diagnostics;
  lines.push("1. Creating WebGPU context... OK");
  if (diagnostics.vendor) {
    lines.push(`adapter.vendor = ${diagnostics.vendor}`);
  }
  if (diagnostics.architecture) {
    lines.push(`adapter.architecture = ${diagnostics.architecture}`);
  }
  log(lines);

  const one = await curve.fr.montOne();
  const two = await curve.fr.add(one, one);
  const three = await curve.fr.add(two, one);
  const threeRegular = await curve.fr.fromMontgomery(three);
  lines.push("2. Field arithmetic... OK");
  lines.push(`fr: 1 + 2 = 0x${shortHex(threeRegular)}`);
  log(lines);

  const nttInput = await buildNTTInput(one, (a, b) => curve.fr.add(a, b), curve.fr.zero());
  const nttForward = await curve.ntt.forward(nttInput);
  const nttRoundTrip = await curve.ntt.inverse(nttForward);
  lines.push("3. NTT round-trip... OK");
  lines.push(`ntt_roundtrip_equal = ${String(equalElements(nttInput, nttRoundTrip))}`);
  log(lines);

  const base = EXAMPLE_BASES[curveId];
  const base2 = await curve.g1.jacobianToAffine(await curve.g1.scalarMulAffine(base, scalarLE(2)));
  const tripleByScalarMul = await curve.g1.jacobianToAffine(await curve.g1.scalarMulAffine(base, scalarLE(3)));
  const tripleByMSM = await curve.g1.jacobianToAffine(await curve.msm.pippengerAffine([base, base2], [scalarLE(1), scalarLE(1)]));
  lines.push("4. G1 scalar mul + MSM... OK");
  lines.push(`g1_scalar_msm_equal = ${String(equalAffine(tripleByScalarMul, tripleByMSM))}`);
  lines.push(`g1_result.x = 0x${shortHex(tripleByScalarMul.x)}`);
  log(lines);

  lines.push("");
  lines.push("PASS: library example completed");
  log(lines);
  context.close();
}

const runButton = mustElement<HTMLButtonElement>("run");
const statusEl = mustElement<HTMLSpanElement>("status");
const logEl = mustElement<HTMLPreElement>("log");
const headingEl = mustElement<HTMLHeadingElement>("heading");
const descriptionEl = mustElement<HTMLParagraphElement>("description");
const curveId = getCurveId();

headingEl.textContent = `${curveLabel(curveId)} CurveGPU Library Example`;
descriptionEl.textContent = "Small consumer-oriented example using the public CurveGPU browser API.";

function writeLog(lines: string[]): void {
  logEl.textContent = lines.join("\n");
}

async function main(): Promise<void> {
  runButton.disabled = true;
  statusEl.textContent = "Running";
  document.body.dataset.status = "running";
  try {
    await runExample(curveId, writeLog);
    statusEl.textContent = "Pass";
    document.body.dataset.status = "pass";
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeLog([`=== ${curveLabel(curveId)} CurveGPU Library Example ===`, "", `FAIL: ${message}`]);
    statusEl.textContent = "Fail";
    document.body.dataset.status = "fail";
    throw error;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void main();
});

if (new URLSearchParams(window.location.search).get("autorun") === "1") {
  void main();
} else {
  writeLog([
    `=== ${curveLabel(curveId)} CurveGPU Library Example ===`,
    "",
    "Press Run to execute a small consumer-oriented example against the public CurveGPU API.",
  ]);
}
