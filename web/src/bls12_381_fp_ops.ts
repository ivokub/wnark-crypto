export {};

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

type Phase3Vectors = {
  element_cases: ElementCase[];
  edge_cases: ElementCase[];
  differential_cases: ElementCase[];
  normalize_cases: NormalizeCase[];
  convert_cases: ConvertCase[];
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

const FP_OP_COPY = 0;
const FP_OP_ZERO = 1;
const FP_OP_ONE = 2;
const FP_OP_ADD = 3;
const FP_OP_SUB = 4;
const FP_OP_NEG = 5;
const FP_OP_DOUBLE = 6;
const FP_OP_NORMALIZE = 7;
const FP_OP_EQUAL = 8;
const FP_OP_MUL = 9;
const FP_OP_SQUARE = 10;
const FP_OP_TO_MONT = 11;
const FP_OP_FROM_MONT = 12;

const FP_ELEMENT_BYTES = 48;
const FP_UNIFORM_BYTES = 32;
const FP_ZERO_HEX = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function setPageState(state: "idle" | "running" | "pass" | "fail"): void {
  document.body.dataset.status = state;
}

function writeLog(lines: string[]): void {
  logEl.textContent = lines.join("\n");
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

function packHexBatch(hexValues: readonly string[]): Uint8Array {
  const out = new Uint8Array(hexValues.length * FP_ELEMENT_BYTES);
  hexValues.forEach((hex, index) => {
    const bytes = hexToBytes(hex);
    if (bytes.length !== FP_ELEMENT_BYTES) {
      throw new Error(`expected ${FP_ELEMENT_BYTES} bytes, got ${bytes.length}`);
    }
    out.set(bytes, index * FP_ELEMENT_BYTES);
  });
  return out;
}

async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors(): Promise<Phase3Vectors> {
  const text = await fetchText("/testdata/vectors/fp/bls12_381_phase3_ops.json?v=1");
  return JSON.parse(text) as Phase3Vectors;
}

async function getAdapterInfo(adapter: GPUAdapter): Promise<GPUAdapterInfo | null> {
  if ("info" in adapter && adapter.info) {
    return adapter.info;
  }
  const compatAdapter = adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
  if (typeof compatAdapter.requestAdapterInfo === "function") {
    try {
      return await compatAdapter.requestAdapterInfo();
    } catch {
      return null;
    }
  }
  return null;
}

async function appendAdapterDiagnostics(adapter: GPUAdapter, lines: string[]): Promise<void> {
  if ("isFallbackAdapter" in adapter) {
    lines.push(`adapter.isFallbackAdapter = ${String(adapter.isFallbackAdapter)}`);
  }
  const info = await getAdapterInfo(adapter);
  if (!info) {
    lines.push("adapter.info = unavailable");
    return;
  }
  if (info.vendor) lines.push(`adapter.vendor = ${info.vendor}`);
  if (info.architecture) lines.push(`adapter.architecture = ${info.architecture}`);
  if (info.device) lines.push(`adapter.device = ${info.device}`);
  if (info.description) lines.push(`adapter.description = ${info.description}`);
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

function createKernel(device: GPUDevice, shaderCode: string): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
  const shaderModule = device.createShaderModule({
    label: "bls12-381-fp-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bls12-381-fp-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bls12-381-fp-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bls12-381-fp-pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "fp_ops_main",
    },
  });
  return { pipeline, bindGroupLayout };
}

async function runOp(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  opcode: number,
  inputAHex: readonly string[],
  inputBHex: readonly string[],
): Promise<string[]> {
  const count = Math.max(inputAHex.length, inputBHex.length);
  const zeros = Array.from({ length: count }, () => FP_ZERO_HEX);
  const aHex = inputAHex.length === 0 ? zeros : inputAHex;
  const bHex = inputBHex.length === 0 ? zeros : inputBHex;
  if (aHex.length !== count || bHex.length !== count) {
    throw new Error("mismatched batch lengths");
  }

  const inputBytes = count * FP_ELEMENT_BYTES;
  const paramsBytes = new Uint8Array(FP_UNIFORM_BYTES);
  const paramsView = new DataView(paramsBytes.buffer);
  paramsView.setUint32(0, count, true);
  paramsView.setUint32(4, opcode, true);

  const inputABuffer = createStorageBuffer(device, "bls12-381-fp-input-a", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, "bls12-381-fp-input-b", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, "bls12-381-fp-output", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readbackBuffer = createStorageBuffer(device, "bls12-381-fp-readback", inputBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = createStorageBuffer(device, "bls12-381-fp-params", FP_UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  device.queue.writeBuffer(inputABuffer, 0, packHexBatch(aHex));
  device.queue.writeBuffer(inputBBuffer, 0, packHexBatch(bHex));
  device.queue.writeBuffer(uniformBuffer, 0, paramsBytes);

  const bindGroup = device.createBindGroup({
    label: "bls12-381-fp-bg",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer, size: inputBytes } },
      { binding: 1, resource: { buffer: inputBBuffer, size: inputBytes } },
      { binding: 2, resource: { buffer: outputBuffer, size: inputBytes } },
      { binding: 3, resource: { buffer: uniformBuffer, size: FP_UNIFORM_BYTES } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "bls12-381-fp-encoder" });
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
    const slice = bytes.slice(i * FP_ELEMENT_BYTES, (i + 1) * FP_ELEMENT_BYTES);
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

function combineElementCases(vectors: Phase3Vectors): ElementCase[] {
  return [...vectors.element_cases, ...vectors.edge_cases, ...vectors.differential_cases];
}

async function runBrowserSmoke(): Promise<string[]> {
  const lines: string[] = ["=== BLS12-381 fp Phase 3 Browser Smoke ===", ""];
  writeLog(lines);

  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser.");
  }

  lines.push("1. Requesting adapter...");
  writeLog(lines);
  const adapter = await withTimeout(navigator.gpu.requestAdapter(), 10000, "requestAdapter");
  if (!adapter) {
    throw new Error("Failed to acquire a WebGPU adapter.");
  }
  lines[lines.length - 1] += " OK";
  await appendAdapterDiagnostics(adapter, lines);
  writeLog(lines);

  lines.push("2. Requesting device...");
  writeLog(lines);
  const device = await withTimeout(adapter.requestDevice(), 10000, "requestDevice");
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  lines.push("3. Loading shader and vectors...");
  writeLog(lines);
  const [shaderCode, vectors] = await Promise.all([
    fetchText("/shaders/curves/bls12_381/fp_arith.wgsl?v=1"),
    fetchVectors(),
  ]);
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  lines.push("4. Creating pipeline...");
  writeLog(lines);
  const kernel = createKernel(device, shaderCode);
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
  const zeros = Array.from({ length: elementCases.length }, () => FP_ZERO_HEX);
  const oneMontHex = mustFindConvertCase(vectors.convert_cases, "one").mont_bytes_le;

  verifyBatch("copy", await runOp(device, kernel, FP_OP_COPY, aHex, bHex), aHex, lines);
  writeLog(lines);
  verifyBatch("equal", await runOp(device, kernel, FP_OP_EQUAL, aHex, bHex), elementCases.map((item) => item.equal_bytes_le), lines);
  writeLog(lines);
  verifyBatch("zero", await runOp(device, kernel, FP_OP_ZERO, zeros, zeros), zeros, lines);
  writeLog(lines);
  verifyBatch("one", await runOp(device, kernel, FP_OP_ONE, zeros, zeros), Array.from({ length: elementCases.length }, () => oneMontHex), lines);
  writeLog(lines);
  verifyBatch("add", await runOp(device, kernel, FP_OP_ADD, aHex, bHex), elementCases.map((item) => item.add_bytes_le), lines);
  writeLog(lines);
  verifyBatch("sub", await runOp(device, kernel, FP_OP_SUB, aHex, bHex), elementCases.map((item) => item.sub_bytes_le), lines);
  writeLog(lines);
  verifyBatch("neg", await runOp(device, kernel, FP_OP_NEG, aHex, zeros), elementCases.map((item) => item.neg_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("double", await runOp(device, kernel, FP_OP_DOUBLE, aHex, zeros), elementCases.map((item) => item.double_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("mul", await runOp(device, kernel, FP_OP_MUL, aHex, bHex), elementCases.map((item) => item.mul_bytes_le), lines);
  writeLog(lines);
  verifyBatch("square", await runOp(device, kernel, FP_OP_SQUARE, aHex, zeros), elementCases.map((item) => item.square_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch(
    "to_mont",
    await runOp(device, kernel, FP_OP_TO_MONT, vectors.convert_cases.map((item) => item.regular_bytes_le), Array.from({ length: vectors.convert_cases.length }, () => FP_ZERO_HEX)),
    vectors.convert_cases.map((item) => item.mont_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "from_mont",
    await runOp(device, kernel, FP_OP_FROM_MONT, vectors.convert_cases.map((item) => item.mont_bytes_le), Array.from({ length: vectors.convert_cases.length }, () => FP_ZERO_HEX)),
    vectors.convert_cases.map((item) => item.regular_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "normalize",
    await runOp(device, kernel, FP_OP_NORMALIZE, vectors.normalize_cases.map((item) => item.input_bytes_le), Array.from({ length: vectors.normalize_cases.length }, () => FP_ZERO_HEX)),
    vectors.normalize_cases.map((item) => item.expected_bytes_le),
    lines,
  );
  writeLog(lines);

  lines.push("");
  lines.push("PASS: BLS12-381 fp browser smoke succeeded");
  writeLog(lines);
  return lines;
}

async function main(): Promise<void> {
  runButton.disabled = true;
  setPageState("running");
  setStatus("Running");

  try {
    const lines = await runBrowserSmoke();
    writeLog(lines);
    setPageState("pass");
    setStatus("Pass");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeLog(["=== BLS12-381 fp Phase 3 Browser Smoke ===", "", `FAIL: ${message}`]);
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

const params = new URLSearchParams(window.location.search);
if (params.get("autorun") === "1") {
  void main();
} else {
  writeLog([
    "=== BLS12-381 fp Phase 3 Browser Smoke ===",
    "",
    "Press Run to execute the BLS12-381 fp smoke test in browser WebGPU.",
  ]);
}
