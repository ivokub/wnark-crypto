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

const FP_ELEMENT_BYTES = 32;
const FP_UNIFORM_BYTES = 32;
const FP_ZERO_HEX = "0000000000000000000000000000000000000000000000000000000000000000";

const runButton = document.getElementById("run");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

function setStatus(text) {
  statusEl.textContent = text;
}

function setPageState(state) {
  document.body.dataset.status = state;
}

function writeLog(lines) {
  logEl.textContent = lines.join("\n");
}

function hexToBytes(hex) {
  if (hex.length % 2 !== 0) {
    throw new Error(`invalid hex length ${hex.length}`);
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function packHexBatch(hexValues) {
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

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors() {
  const text = await fetchText("/testdata/vectors/fp/bn254_phase3_ops.json");
  return JSON.parse(text);
}

async function getAdapterInfo(adapter) {
  if (adapter.info) {
    return adapter.info;
  }
  if (typeof adapter.requestAdapterInfo === "function") {
    try {
      return await adapter.requestAdapterInfo();
    } catch {
      return null;
    }
  }
  return null;
}

async function appendAdapterDiagnostics(adapter, lines) {
  if ("isFallbackAdapter" in adapter) {
    lines.push(`adapter.isFallbackAdapter = ${String(adapter.isFallbackAdapter)}`);
  }
  const info = await getAdapterInfo(adapter);
  if (!info) {
    lines.push("adapter.info = unavailable");
    return;
  }
  if (info.vendor) {
    lines.push(`adapter.vendor = ${info.vendor}`);
  }
  if (info.architecture) {
    lines.push(`adapter.architecture = ${info.architecture}`);
  }
  if (info.device) {
    lines.push(`adapter.device = ${info.device}`);
  }
  if (info.description) {
    lines.push(`adapter.description = ${info.description}`);
  }
}

function createStorageBuffer(device, label, size, usage) {
  return device.createBuffer({ label, size, usage });
}

async function withTimeout(promise, ms, label) {
  let timeoutId = 0;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeoutId = window.setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    if (timeoutId !== 0) {
      window.clearTimeout(timeoutId);
    }
  }
}

function createKernel(device, shaderCode) {
  const shaderModule = device.createShaderModule({
    label: "bn254-fp-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-fp-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bn254-fp-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-fp-pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "fp_ops_main",
    },
  });
  return { pipeline, bindGroupLayout };
}

async function runOp(device, kernel, opcode, inputAHex, inputBHex) {
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

  const inputABuffer = createStorageBuffer(device, "bn254-fp-input-a", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, "bn254-fp-input-b", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, "bn254-fp-output", inputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readbackBuffer = createStorageBuffer(device, "bn254-fp-readback", inputBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = createStorageBuffer(device, "bn254-fp-params", FP_UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  device.queue.writeBuffer(inputABuffer, 0, packHexBatch(aHex));
  device.queue.writeBuffer(inputBBuffer, 0, packHexBatch(bHex));
  device.queue.writeBuffer(uniformBuffer, 0, paramsBytes);

  const bindGroup = device.createBindGroup({
    label: "bn254-fp-bg",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer, size: inputBytes } },
      { binding: 1, resource: { buffer: inputBBuffer, size: inputBytes } },
      { binding: 2, resource: { buffer: outputBuffer, size: inputBytes } },
      { binding: 3, resource: { buffer: uniformBuffer, size: FP_UNIFORM_BYTES } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "bn254-fp-encoder" });
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

  const out = [];
  for (let i = 0; i < count; i += 1) {
    const slice = bytes.slice(i * FP_ELEMENT_BYTES, (i + 1) * FP_ELEMENT_BYTES);
    out.push(bytesToHex(slice));
  }
  return out;
}

function verifyBatch(name, got, expected, lines) {
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

function mustFindConvertCase(cases, name) {
  const found = cases.find((item) => item.name === name);
  if (!found) {
    throw new Error(`missing convert case ${name}`);
  }
  return found;
}

function combineElementCases(vectors) {
  return [
    ...vectors.element_cases,
    ...vectors.edge_cases,
    ...vectors.differential_cases,
  ];
}

async function runBrowserSmoke() {
  const lines = ["=== BN254 fp Phase 3 Browser Smoke ===", ""];
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
    fetchText("/shaders/curves/bn254/fp_arith.wgsl"),
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
    await runOp(
      device,
      kernel,
      FP_OP_TO_MONT,
      vectors.convert_cases.map((item) => item.regular_bytes_le),
      Array.from({ length: vectors.convert_cases.length }, () => FP_ZERO_HEX),
    ),
    vectors.convert_cases.map((item) => item.mont_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "from_mont",
    await runOp(
      device,
      kernel,
      FP_OP_FROM_MONT,
      vectors.convert_cases.map((item) => item.mont_bytes_le),
      Array.from({ length: vectors.convert_cases.length }, () => FP_ZERO_HEX),
    ),
    vectors.convert_cases.map((item) => item.regular_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "normalize",
    await runOp(
      device,
      kernel,
      FP_OP_NORMALIZE,
      vectors.normalize_cases.map((item) => item.input_bytes_le),
      Array.from({ length: vectors.normalize_cases.length }, () => FP_ZERO_HEX),
    ),
    vectors.normalize_cases.map((item) => item.expected_bytes_le),
    lines,
  );
  writeLog(lines);

  lines.push("");
  lines.push("PASS: BN254 fp browser smoke succeeded");
  writeLog(lines);
  return lines;
}

async function main() {
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
    writeLog(["=== BN254 fp Phase 3 Browser Smoke ===", "", `FAIL: ${message}`]);
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
    "=== BN254 fp Phase 3 Browser Smoke ===",
    "",
    "Press Run to execute the BN254 fp smoke test in browser WebGPU.",
  ]);
}
