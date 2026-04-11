import { loadFieldKernel } from "./curvegpu/kernels.js";

async function main(): Promise<void> {
  const log = document.getElementById("log");
  if (!log) {
    throw new Error("missing log element");
  }
  const kernel = await loadFieldKernel("bn254", "fr");
  log.textContent = [
    "CurveGPU BN254 scaffold",
    `shader: ${kernel.shaderPath}`,
    `hostWords: ${kernel.shape.hostWords}`,
    `gpuLimbs: ${kernel.shape.gpuLimbs}`,
  ].join("\n");
}

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  const log = document.getElementById("log");
  if (log) {
    log.textContent = `ERROR: ${message}`;
  }
});
