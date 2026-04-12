import {
  createCurveGPUContext,
  createCurveModule,
  type CurveGPUContext,
  type CurveModule,
  type SupportedCurveID,
} from "./index.js";

export function getRequestedCurveId(search = window.location.search): SupportedCurveID {
  const curve = new URLSearchParams(search).get("curve") ?? "bn254";
  if (curve !== "bn254" && curve !== "bls12_381") {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return curve;
}

export function curveDisplayName(curve: SupportedCurveID): string {
  return curve === "bn254" ? "BN254" : "BLS12-381";
}

export function appendContextDiagnostics(lines: string[], context: CurveGPUContext): void {
  const diagnostics = context.diagnostics;
  if (diagnostics.isFallbackAdapter !== undefined) {
    lines.push(`adapter.isFallbackAdapter = ${String(diagnostics.isFallbackAdapter)}`);
  }
  if (diagnostics.vendor) {
    lines.push(`adapter.vendor = ${diagnostics.vendor}`);
  }
  if (diagnostics.architecture) {
    lines.push(`adapter.architecture = ${diagnostics.architecture}`);
  }
  if (!diagnostics.vendor && !diagnostics.architecture) {
    lines.push("adapter.info = unavailable");
  }
}

export async function createRequestedCurveModule(curve = getRequestedCurveId()): Promise<CurveModule> {
  const context = await createCurveGPUContext();
  return createCurveModule(context, curve);
}
