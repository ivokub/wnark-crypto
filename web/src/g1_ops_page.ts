export {};

import {
  appendAdapterDiagnostics,
  bytesToHex,
  createPageUI,
  fetchText,
  hexToBytes,
} from "./curvegpu/browser_utils.js";
import { fetchShaderParts } from "./curvegpu/shaders.js";

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
  curve: string;
  title: string;
  vectorPath: string;
  shaderParts: string[];
  labelPrefix: string;
  fpBytes: number;
  pointBytes: number;
  zeroHex: string;
};

declare const GPUShaderStage: { COMPUTE: number };
declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  MAP_READ: number;
  UNIFORM: number;
};
declare const GPUMapMode: { READ: number };

const G1_OP_COPY = 0;
const G1_OP_JAC_INFINITY = 1;
const G1_OP_AFFINE_TO_JAC = 2;
const G1_OP_NEG_JAC = 3;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;

const UNIFORM_BYTES = 32;

const CONFIGS: Record<string, G1OpsConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G1 Ops Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bn254_g1_ops.json",
    shaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl?v=2#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl?v=1",
    ],
    labelPrefix: "bn254-g1",
    fpBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G1 Ops Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bls12_381_g1_ops.json?v=1",
    shaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=3#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-core",
      "/shaders/curves/bls12_381/g1_arith.wgsl?v=2",
    ],
    labelPrefix: "bls12-381-g1",
    fpBytes: 48,
    pointBytes: 144,
    zeroHex: "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): G1OpsConfig {
  const curve = new URLSearchParams(window.location.search).get("curve") ?? "bn254";
  const config = CONFIGS[curve];
  if (!config) {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return config;
}

async function fetchVectors(config: G1OpsConfig): Promise<G1OpsVectors> {
  const text = await fetchText(config.vectorPath);
  return JSON.parse(text) as G1OpsVectors;
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function createKernel(device: GPUDevice, shaderCode: string, config: G1OpsConfig): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
  const shaderModule = device.createShaderModule({
    label: `${config.labelPrefix}-shader`,
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${config.labelPrefix}-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${config.labelPrefix}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${config.labelPrefix}-pipeline`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "g1_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function findOneMontZ(vectors: G1OpsVectors, zeroHex: string): string {
  for (const pointCase of vectors.point_cases) {
    if (pointCase.p_jacobian.z_bytes_le !== zeroHex) {
      return pointCase.p_jacobian.z_bytes_le;
    }
  }
  throw new Error("missing non-zero affine flag in vectors");
}

function affineToKernelPoint(point: AffinePoint, oneMontZ: string, zeroHex: string): JacobianPoint {
  const isInfinity = point.x_bytes_le === zeroHex && point.y_bytes_le === zeroHex;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? zeroHex : oneMontZ,
  };
}

function packPointBatch(points: readonly JacobianPoint[], config: G1OpsConfig): Uint8Array {
  const out = new Uint8Array(points.length * config.pointBytes);
  points.forEach((point, index) => {
    const base = index * config.pointBytes;
    out.set(hexToBytes(point.x_bytes_le), base);
    out.set(hexToBytes(point.y_bytes_le), base + config.fpBytes);
    out.set(hexToBytes(point.z_bytes_le), base + 2 * config.fpBytes);
  });
  return out;
}

function unpackPointBatch(bytes: Uint8Array, count: number, config: G1OpsConfig): JacobianPoint[] {
  const out: JacobianPoint[] = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * config.pointBytes;
    out.push({
      x_bytes_le: bytesToHex(bytes.slice(base, base + config.fpBytes)),
      y_bytes_le: bytesToHex(bytes.slice(base + config.fpBytes, base + 2 * config.fpBytes)),
      z_bytes_le: bytesToHex(bytes.slice(base + 2 * config.fpBytes, base + 3 * config.fpBytes)),
    });
  }
  return out;
}

function expectPointBatch(name: string, got: readonly JacobianPoint[], want: readonly JacobianPoint[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i].x_bytes_le !== want[i].x_bytes_le || got[i].y_bytes_le !== want[i].y_bytes_le || got[i].z_bytes_le !== want[i].z_bytes_le) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
}

async function runOp(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  config: G1OpsConfig,
  opcode: number,
  inputA: readonly JacobianPoint[],
  inputB: readonly JacobianPoint[],
): Promise<JacobianPoint[]> {
  const count = inputA.length;
  const aBytes = packPointBatch(inputA, config);
  const bBytes = packPointBatch(inputB, config);
  const byteSize = count * config.pointBytes;

  const inputABuffer = createStorageBuffer(device, `${config.labelPrefix}-input-a`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, `${config.labelPrefix}-input-b`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, `${config.labelPrefix}-output`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const stagingBuffer = createStorageBuffer(device, `${config.labelPrefix}-staging`, byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = device.createBuffer({
    label: `${config.labelPrefix}-params`,
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputABuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  device.queue.writeBuffer(inputBBuffer, 0, bBytes.buffer.slice(bBytes.byteOffset, bBytes.byteOffset + bBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    label: `${config.labelPrefix}-bind-group`,
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer } },
      { binding: 1, resource: { buffer: inputBBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: `${config.labelPrefix}-encoder` });
  const pass = encoder.beginComputePass({ label: `${config.labelPrefix}-pass` });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  inputABuffer.destroy();
  inputBBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();

  return unpackPointBatch(result, count, config);
}

async function runSmoke(config: G1OpsConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    lines.push("1. Requesting adapter... OK");
    await appendAdapterDiagnostics(adapter, lines);

    const device = await adapter.requestDevice();
    lines.push("2. Requesting device... OK");

    const [shaderText, vectors] = await Promise.all([fetchShaderParts(config.shaderParts), fetchVectors(config)]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`cases.g1 = ${vectors.point_cases.length}`);

    const kernel = createKernel(device, shaderText, config);
    lines.push("4. Creating pipeline... OK");

    const oneMontZ = findOneMontZ(vectors, config.zeroHex);
    const pAffineInputs = vectors.point_cases.map((pointCase) => affineToKernelPoint(pointCase.p_affine, oneMontZ, config.zeroHex));
    const qAffineInputs = vectors.point_cases.map((pointCase) => affineToKernelPoint(pointCase.q_affine, oneMontZ, config.zeroHex));
    const pJac = vectors.point_cases.map((pointCase) => pointCase.p_jacobian);
    const negWant = vectors.point_cases.map((pointCase) => pointCase.neg_p_jacobian);
    const doubleWant = vectors.point_cases.map((pointCase) => pointCase.double_p_jacobian);
    const addWant = vectors.point_cases.map((pointCase) => pointCase.add_mixed_p_plus_q_jacobian);
    const affineWant = vectors.point_cases.map((pointCase) => pointCase.p_affine_output);
    const affineAddWant = vectors.point_cases.map((pointCase) => pointCase.affine_add_p_plus_q);
    const infinityWant = vectors.point_cases.map(() => ({ x_bytes_le: oneMontZ, y_bytes_le: oneMontZ, z_bytes_le: config.zeroHex }));
    const zeros = vectors.point_cases.map(() => ({ x_bytes_le: config.zeroHex, y_bytes_le: config.zeroHex, z_bytes_le: config.zeroHex }));

    expectPointBatch("copy", await runOp(device, kernel, config, G1_OP_COPY, pJac, zeros), pJac);
    lines.push("copy: OK");

    expectPointBatch("jac_infinity", await runOp(device, kernel, config, G1_OP_JAC_INFINITY, zeros, zeros), infinityWant);
    lines.push("jac_infinity: OK");

    expectPointBatch("affine_to_jac", await runOp(device, kernel, config, G1_OP_AFFINE_TO_JAC, pAffineInputs, zeros), pJac);
    lines.push("affine_to_jac: OK");

    expectPointBatch("neg_jac", await runOp(device, kernel, config, G1_OP_NEG_JAC, pJac, zeros), negWant);
    lines.push("neg_jac: OK");

    expectPointBatch("jac_to_affine", await runOp(device, kernel, config, G1_OP_JAC_TO_AFFINE, pJac, zeros), affineWant);
    lines.push("jac_to_affine: OK");

    expectPointBatch("double_jac", await runOp(device, kernel, config, G1_OP_DOUBLE_JAC, pJac, zeros), doubleWant);
    lines.push("double_jac: OK");

    expectPointBatch("add_mixed", await runOp(device, kernel, config, G1_OP_ADD_MIXED, pJac, qAffineInputs), addWant);
    lines.push("add_mixed: OK");

    expectPointBatch("affine_add", await runOp(device, kernel, config, G1_OP_AFFINE_ADD, pAffineInputs, qAffineInputs), affineAddWant);
    lines.push("affine_add: OK");

    lines.push("");
    lines.push(`PASS: ${config.curve === "bn254" ? "BN254" : "BLS12-381"} G1 Ops browser smoke succeeded`);
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
