export {};

import {
  appendAdapterDiagnostics,
  bytesToHex,
  createPageUI,
  fetchText,
  hexToBytes,
} from "./curvegpu/browser_utils.js";

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

type FROpsVectors = {
  element_cases: ElementCase[];
  edge_cases: ElementCase[];
  differential_cases: ElementCase[];
  normalize_cases: NormalizeCase[];
  convert_cases: ConvertCase[];
};

type FrOpsConfig = {
  curve: string;
  title: string;
  vectorPath: string;
  shaderPath: string;
  labelPrefix: string;
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

const FR_OP_COPY = 0;
const FR_OP_ZERO = 1;
const FR_OP_ONE = 2;
const FR_OP_ADD = 3;
const FR_OP_SUB = 4;
const FR_OP_NEG = 5;
const FR_OP_DOUBLE = 6;
const FR_OP_NORMALIZE = 7;
const FR_OP_EQUAL = 8;
const FR_OP_MUL = 9;
const FR_OP_SQUARE = 10;
const FR_OP_TO_MONT = 11;
const FR_OP_FROM_MONT = 12;

const FR_ELEMENT_BYTES = 32;
const FR_UNIFORM_BYTES = 32;
const FR_ZERO_HEX = "0000000000000000000000000000000000000000000000000000000000000000";

const CONFIGS: Record<string, FrOpsConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 fr Ops Browser Smoke",
    vectorPath: "/testdata/vectors/fr/bn254_fr_ops.json",
    shaderPath: "/shaders/curves/bn254/fr_arith.wgsl",
    labelPrefix: "bn254-fr",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 fr Ops Browser Smoke",
    vectorPath: "/testdata/vectors/fr/bls12_381_fr_ops.json?v=2",
    shaderPath: "/shaders/curves/bls12_381/fr_arith.wgsl?v=2",
    labelPrefix: "bls12-381-fr",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function getConfig(): FrOpsConfig {
  const curve = new URLSearchParams(window.location.search).get("curve") ?? "bn254";
  const config = CONFIGS[curve];
  if (!config) {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return config;
}

function packHexBatch(hexValues: readonly string[]): Uint8Array {
  const out = new Uint8Array(hexValues.length * FR_ELEMENT_BYTES);
  hexValues.forEach((hex, index) => {
    const bytes = hexToBytes(hex);
    if (bytes.length !== FR_ELEMENT_BYTES) {
      throw new Error(`expected ${FR_ELEMENT_BYTES} bytes, got ${bytes.length}`);
    }
    out.set(bytes, index * FR_ELEMENT_BYTES);
  });
  return out;
}

async function fetchVectors(config: FrOpsConfig): Promise<FROpsVectors> {
  const text = await fetchText(config.vectorPath);
  return JSON.parse(text) as FROpsVectors;
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timeoutId = 0;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timeoutId = window.setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    if (timeoutId !== 0) {
      window.clearTimeout(timeoutId);
    }
  }
}

function createKernel(device: GPUDevice, shaderCode: string, config: FrOpsConfig): {
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
    compute: {
      module: shaderModule,
      entryPoint: "fr_ops_main",
    },
  });
  return { pipeline, bindGroupLayout };
}

async function runOp(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  config: FrOpsConfig,
  opcode: number,
  inputAHex: readonly string[],
  inputBHex: readonly string[],
): Promise<string[]> {
  const count = Math.max(inputAHex.length, inputBHex.length);
  const zeros = Array.from({ length: count }, () => FR_ZERO_HEX);
  const aHex = inputAHex.length === 0 ? zeros : inputAHex;
  const bHex = inputBHex.length === 0 ? zeros : inputBHex;
  if (aHex.length !== count || bHex.length !== count) {
    throw new Error("mismatched batch lengths");
  }

  const inputBytes = count * FR_ELEMENT_BYTES;
  const paramsBytes = new Uint8Array(FR_UNIFORM_BYTES);
  const paramsView = new DataView(paramsBytes.buffer);
  paramsView.setUint32(0, count, true);
  paramsView.setUint32(4, opcode, true);

  const inputABuffer = createStorageBuffer(device, `${config.labelPrefix}-input-a`, inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, `${config.labelPrefix}-input-b`, inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, `${config.labelPrefix}-output`, inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readbackBuffer = createStorageBuffer(device, `${config.labelPrefix}-readback`, inputBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = createStorageBuffer(device, `${config.labelPrefix}-params`, FR_UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  device.queue.writeBuffer(inputABuffer, 0, packHexBatch(aHex));
  device.queue.writeBuffer(inputBBuffer, 0, packHexBatch(bHex));
  device.queue.writeBuffer(uniformBuffer, 0, paramsBytes);

  const bindGroup = device.createBindGroup({
    label: `${config.labelPrefix}-bg`,
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer, size: inputBytes } },
      { binding: 1, resource: { buffer: inputBBuffer, size: inputBytes } },
      { binding: 2, resource: { buffer: outputBuffer, size: inputBytes } },
      { binding: 3, resource: { buffer: uniformBuffer, size: FR_UNIFORM_BYTES } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: `${config.labelPrefix}-encoder` });
  const pass = encoder.beginComputePass();
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, inputBytes);
  device.queue.submit([encoder.finish()]);

  await withTimeout(device.queue.onSubmittedWorkDone(), 10000, `queue completion for op ${opcode}`);
  await withTimeout(readbackBuffer.mapAsync(GPUMapMode.READ), 10000, `mapAsync for op ${opcode}`);
  const mapped = readbackBuffer.getMappedRange();
  const bytes = new Uint8Array(mapped.slice(0));
  readbackBuffer.unmap();

  const out: string[] = [];
  for (let i = 0; i < count; i += 1) {
    const slice = bytes.slice(i * FR_ELEMENT_BYTES, (i + 1) * FR_ELEMENT_BYTES);
    out.push(bytesToHex(slice));
  }
  return out;
}

function verifyBatch(name: string, got: readonly string[], expected: readonly string[], lines: string[]): void {
  if (got.length !== expected.length) {
    throw new Error(`${name}: result count mismatch ${got.length} != ${expected.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i] !== expected[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${got[i]} want=${expected[i]}`);
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

function combineElementCases(vectors: FROpsVectors): ElementCase[] {
  return [...vectors.element_cases, ...vectors.edge_cases, ...vectors.differential_cases];
}

async function runBrowserSmoke(config: FrOpsConfig): Promise<string[]> {
  const lines: string[] = [`=== ${config.title} ===`, ""];
  writeLog(lines);

  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser.");
  }

  lines.push("1. Loading shader and vectors...");
  writeLog(lines);
  const [shaderCode, vectors] = await Promise.all([fetchText(config.shaderPath), fetchVectors(config)]);
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  lines.push("2. Requesting adapter...");
  writeLog(lines);
  const adapter = await withTimeout(navigator.gpu.requestAdapter(), 10000, "requestAdapter");
  if (!adapter) {
    throw new Error("Failed to acquire a WebGPU adapter.");
  }
  lines[lines.length - 1] += " OK";
  await appendAdapterDiagnostics(adapter, lines);
  writeLog(lines);

  lines.push("3. Requesting device...");
  writeLog(lines);
  const device = await withTimeout(adapter.requestDevice(), 10000, "requestDevice");
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  lines.push("4. Creating pipeline...");
  writeLog(lines);
  const kernel = createKernel(device, shaderCode, config);
  lines[lines.length - 1] += " OK";
  lines.push(`cases.sanity = ${vectors.element_cases.length}`);
  lines.push(`cases.edge = ${vectors.edge_cases.length}`);
  lines.push(`cases.differential = ${vectors.differential_cases.length}`);
  lines.push(`cases.normalize = ${vectors.normalize_cases.length}`);
  lines.push(`cases.convert = ${vectors.convert_cases.length}`);
  writeLog(lines);

  const elementCases = combineElementCases(vectors);
  const aHex = elementCases.map((item) => item.a_bytes_le);
  const bHex = elementCases.map((item) => item.b_bytes_le);
  const zeros = Array.from({ length: elementCases.length }, () => FR_ZERO_HEX);
  const oneMontHex = mustFindConvertCase(vectors.convert_cases, "one").mont_bytes_le;

  verifyBatch("copy", await runOp(device, kernel, config, FR_OP_COPY, aHex, bHex), aHex, lines);
  writeLog(lines);
  verifyBatch("equal", await runOp(device, kernel, config, FR_OP_EQUAL, aHex, bHex), elementCases.map((item) => item.equal_bytes_le), lines);
  writeLog(lines);
  verifyBatch("zero", await runOp(device, kernel, config, FR_OP_ZERO, zeros, zeros), zeros, lines);
  writeLog(lines);
  verifyBatch("one", await runOp(device, kernel, config, FR_OP_ONE, zeros, zeros), Array.from({ length: elementCases.length }, () => oneMontHex), lines);
  writeLog(lines);
  verifyBatch("add", await runOp(device, kernel, config, FR_OP_ADD, aHex, bHex), elementCases.map((item) => item.add_bytes_le), lines);
  writeLog(lines);
  verifyBatch("sub", await runOp(device, kernel, config, FR_OP_SUB, aHex, bHex), elementCases.map((item) => item.sub_bytes_le), lines);
  writeLog(lines);
  verifyBatch("neg", await runOp(device, kernel, config, FR_OP_NEG, aHex, zeros), elementCases.map((item) => item.neg_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("double", await runOp(device, kernel, config, FR_OP_DOUBLE, aHex, zeros), elementCases.map((item) => item.double_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("mul", await runOp(device, kernel, config, FR_OP_MUL, aHex, bHex), elementCases.map((item) => item.mul_bytes_le), lines);
  writeLog(lines);
  verifyBatch("square", await runOp(device, kernel, config, FR_OP_SQUARE, aHex, zeros), elementCases.map((item) => item.square_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch(
    "to_mont",
    await runOp(device, kernel, config, FR_OP_TO_MONT, vectors.convert_cases.map((item) => item.regular_bytes_le), Array.from({ length: vectors.convert_cases.length }, () => FR_ZERO_HEX)),
    vectors.convert_cases.map((item) => item.mont_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "from_mont",
    await runOp(device, kernel, config, FR_OP_FROM_MONT, vectors.convert_cases.map((item) => item.mont_bytes_le), Array.from({ length: vectors.convert_cases.length }, () => FR_ZERO_HEX)),
    vectors.convert_cases.map((item) => item.regular_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "normalize",
    await runOp(device, kernel, config, FR_OP_NORMALIZE, vectors.normalize_cases.map((item) => item.input_bytes_le), Array.from({ length: vectors.normalize_cases.length }, () => FR_ZERO_HEX)),
    vectors.normalize_cases.map((item) => item.expected_bytes_le),
    lines,
  );
  writeLog(lines);

  lines.push("");
  lines.push(`PASS: ${config.curve === "bn254" ? "BN254" : "BLS12-381"} fr browser smoke succeeded`);
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
  writeLog([`=== ${config.title} ===`, "", `Press Run to execute the ${config.curve} fr smoke test in browser WebGPU.`]);
}
