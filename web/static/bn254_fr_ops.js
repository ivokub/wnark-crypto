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

async function requestAdapterWithRetry(attempts, ms) {
  let lastError = null;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      const adapter = await withTimeout(navigator.gpu.requestAdapter(), ms, `requestAdapter attempt ${i}`);
      if (adapter) {
        return adapter;
      }
      lastError = new Error(`requestAdapter attempt ${i} returned null`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error("requestAdapter failed");
}

async function requestDeviceWithRetry(adapter, attempts, ms) {
  let lastError = null;
  for (let i = 1; i <= attempts; i += 1) {
    try {
      return await withTimeout(adapter.requestDevice(), ms, `requestDevice attempt ${i}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error("requestDevice failed");
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

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors() {
  const text = await fetchText("/testdata/vectors/fr/bn254_phase2_ops.json");
  return JSON.parse(text);
}

function createStorageBuffer(device, label, size, usage) {
  return device.createBuffer({ label, size, usage });
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

function specializeShaderForLimbs(shaderCode, limbCount) {
  return shaderCode
    .replace("var<storage, read> input_a: array<u32>;", `var<storage, read> input_a: array<u32, ${limbCount}>;`)
    .replace("var<storage, read> input_b: array<u32>;", `var<storage, read> input_b: array<u32, ${limbCount}>;`)
    .replace("var<storage, read_write> output: array<u32>;", `var<storage, read_write> output: array<u32, ${limbCount}>;`);
}

function trivialShaderForOp(opcode, limbCount) {
  if (opcode === FR_OP_COPY) {
    return `
struct Params { count: u32, _pad0: vec3<u32>, }
@group(0) @binding(0) var<storage, read> input_a: array<u32, ${limbCount}>;
@group(0) @binding(1) var<storage, read> input_b: array<u32, ${limbCount}>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, ${limbCount}>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn fr_ops_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) { return; }
  let base = i * 8u;
  output[base + 0u] = input_a[base + 0u];
  output[base + 1u] = input_a[base + 1u];
  output[base + 2u] = input_a[base + 2u];
  output[base + 3u] = input_a[base + 3u];
  output[base + 4u] = input_a[base + 4u];
  output[base + 5u] = input_a[base + 5u];
  output[base + 6u] = input_a[base + 6u];
  output[base + 7u] = input_a[base + 7u];
}`;
  }
  if (opcode === FR_OP_ZERO) {
    return `
struct Params { count: u32, _pad0: vec3<u32>, }
@group(0) @binding(0) var<storage, read> input_a: array<u32, ${limbCount}>;
@group(0) @binding(1) var<storage, read> input_b: array<u32, ${limbCount}>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, ${limbCount}>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn fr_ops_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) { return; }
  let base = i * 8u;
  output[base + 0u] = 0u;
  output[base + 1u] = 0u;
  output[base + 2u] = 0u;
  output[base + 3u] = 0u;
  output[base + 4u] = 0u;
  output[base + 5u] = 0u;
  output[base + 6u] = 0u;
  output[base + 7u] = 0u;
}`;
  }
  if (opcode === FR_OP_ONE) {
    return `
struct Params { count: u32, _pad0: vec3<u32>, }
@group(0) @binding(0) var<storage, read> input_a: array<u32, ${limbCount}>;
@group(0) @binding(1) var<storage, read> input_b: array<u32, ${limbCount}>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, ${limbCount}>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn fr_ops_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) { return; }
  let base = i * 8u;
  output[base + 0u] = 0x4ffffffbu;
  output[base + 1u] = 0xac96341cu;
  output[base + 2u] = 0x9f60cd29u;
  output[base + 3u] = 0x36fc7695u;
  output[base + 4u] = 0x7879462eu;
  output[base + 5u] = 0x666ea36fu;
  output[base + 6u] = 0x9a07df2fu;
  output[base + 7u] = 0x0e0a77c1u;
}`;
  }
  if (opcode === FR_OP_EQUAL) {
    return `
struct Params { count: u32, _pad0: vec3<u32>, }
@group(0) @binding(0) var<storage, read> input_a: array<u32, ${limbCount}>;
@group(0) @binding(1) var<storage, read> input_b: array<u32, ${limbCount}>;
@group(0) @binding(2) var<storage, read_write> output: array<u32, ${limbCount}>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn fr_ops_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) { return; }
  let base = i * 8u;
  let diff =
    (input_a[base + 0u] ^ input_b[base + 0u]) |
    (input_a[base + 1u] ^ input_b[base + 1u]) |
    (input_a[base + 2u] ^ input_b[base + 2u]) |
    (input_a[base + 3u] ^ input_b[base + 3u]) |
    (input_a[base + 4u] ^ input_b[base + 4u]) |
    (input_a[base + 5u] ^ input_b[base + 5u]) |
    (input_a[base + 6u] ^ input_b[base + 6u]) |
    (input_a[base + 7u] ^ input_b[base + 7u]);
  if (diff == 0u) {
    output[base + 0u] = 0x4ffffffbu;
    output[base + 1u] = 0xac96341cu;
    output[base + 2u] = 0x9f60cd29u;
    output[base + 3u] = 0x36fc7695u;
    output[base + 4u] = 0x7879462eu;
    output[base + 5u] = 0x666ea36fu;
    output[base + 6u] = 0x9a07df2fu;
    output[base + 7u] = 0x0e0a77c1u;
    return;
  }
  output[base + 0u] = 0u;
  output[base + 1u] = 0u;
  output[base + 2u] = 0u;
  output[base + 3u] = 0u;
  output[base + 4u] = 0u;
  output[base + 5u] = 0u;
  output[base + 6u] = 0u;
  output[base + 7u] = 0u;
}`;
  }
  return null;
}

function createKernel(device, shaderCode, limbCount, opcode) {
  const specializedShader = trivialShaderForOp(opcode, limbCount) ?? specializeShaderForLimbs(shaderCode, limbCount);
  const shaderModule = device.createShaderModule({
    label: "bn254-fr-shader",
    code: specializedShader,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-fr-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bn254-fr-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-fr-pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "fr_ops_main",
    },
  });
  return { pipeline, bindGroupLayout };
}

async function runOp(device, kernel, opcode, inputAHex, inputBHex) {
  const count = Math.max(inputAHex.length, inputBHex.length);
  const zeros = Array.from({ length: count }, () => FR_ZERO_HEX);
  const aHex = inputAHex.length === 0 ? zeros : inputAHex;
  const bHex = inputBHex.length === 0 ? zeros : inputBHex;

  const inputBytes = count * FR_ELEMENT_BYTES;
  const paddedBytes = Math.max(256, Math.ceil(inputBytes / 256) * 256);
  const paramsBytes = new Uint8Array(FR_UNIFORM_BYTES);
  const paramsView = new DataView(paramsBytes.buffer);
  paramsView.setUint32(0, count, true);
  paramsView.setUint32(4, opcode, true);

  const inputABuffer = createStorageBuffer(device, "bn254-fr-input-a", paddedBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, "bn254-fr-input-b", paddedBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, "bn254-fr-output", paddedBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readbackBuffer = createStorageBuffer(device, "bn254-fr-readback", paddedBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = createStorageBuffer(device, "bn254-fr-params", FR_UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  device.queue.writeBuffer(inputABuffer, 0, packHexBatch(aHex));
  device.queue.writeBuffer(inputBBuffer, 0, packHexBatch(bHex));
  device.queue.writeBuffer(uniformBuffer, 0, paramsBytes);

  const bindGroup = device.createBindGroup({
    label: "bn254-fr-bg",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer, size: paddedBytes } },
      { binding: 1, resource: { buffer: inputBBuffer, size: paddedBytes } },
      { binding: 2, resource: { buffer: outputBuffer, size: paddedBytes } },
      { binding: 3, resource: { buffer: uniformBuffer, size: FR_UNIFORM_BYTES } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "bn254-fr-encoder" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, paddedBytes);
  device.queue.submit([encoder.finish()]);

  await withTimeout(readbackBuffer.mapAsync(GPUMapMode.READ), 10000, `mapAsync for op ${opcode}`);
  const mapped = readbackBuffer.getMappedRange();
  const bytes = new Uint8Array(mapped.slice(0));
  readbackBuffer.unmap();

  const out = [];
  for (let i = 0; i < count; i += 1) {
    const slice = bytes.slice(i * FR_ELEMENT_BYTES, (i + 1) * FR_ELEMENT_BYTES);
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
  const lines = ["=== BN254 fr Phase 2 Browser Smoke ===", ""];
  writeLog(lines);

  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser.");
  }

  lines.push("1. Requesting adapter...");
  writeLog(lines);
  const adapter = await requestAdapterWithRetry(3, 30000);
  if (!adapter) {
    throw new Error("Failed to acquire a WebGPU adapter.");
  }
  lines[lines.length - 1] += " OK";
  await appendAdapterDiagnostics(adapter, lines);
  writeLog(lines);

  lines.push("2. Requesting device...");
  writeLog(lines);
  const device = await requestDeviceWithRetry(adapter, 3, 30000);
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  device.lost.then((info) => {
    lines.push(`device lost: ${info.message || "unknown"}`);
    writeLog(lines);
  });
  device.addEventListener("uncapturederror", (event) => {
    lines.push(`uncapturederror: ${event.error?.message || String(event.error)}`);
    writeLog(lines);
  });

  lines.push("3. Loading shader and vectors...");
  writeLog(lines);
  const [shaderCode, vectors] = await Promise.all([
    fetchText("/shaders/curves/bn254/fr_arith.wgsl"),
    fetchVectors(),
  ]);
  lines[lines.length - 1] += " OK";
  writeLog(lines);

  lines.push("4. Creating pipeline...");
  writeLog(lines);
  const elementCases = combineElementCases(vectors);
  const aHex = elementCases.map((item) => item.a_bytes_le);
  const bHex = elementCases.map((item) => item.b_bytes_le);
  const zeros = Array.from({ length: elementCases.length }, () => FR_ZERO_HEX);
  const elementLimbCount = aHex.length * 8;
  const normalizeLimbCount = vectors.normalize_cases.length * 8;
  const convertLimbCount = vectors.convert_cases.length * 8;
  const oneMontHex = mustFindConvertCase(vectors.convert_cases, "one").mont_bytes_le;
  lines[lines.length - 1] += " OK";
  lines.push(`cases.sanity = ${vectors.element_cases.length}`);
  lines.push(`cases.edge = ${vectors.edge_cases.length}`);
  lines.push(`cases.differential = ${vectors.differential_cases.length}`);
  lines.push(`cases.normalize = ${vectors.normalize_cases.length}`);
  lines.push(`cases.convert = ${vectors.convert_cases.length}`);
  writeLog(lines);

  verifyBatch("copy", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_COPY), FR_OP_COPY, aHex, bHex), aHex, lines);
  writeLog(lines);
  verifyBatch("equal", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_EQUAL), FR_OP_EQUAL, aHex, bHex), elementCases.map((item) => item.equal_bytes_le), lines);
  writeLog(lines);
  verifyBatch("zero", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_ZERO), FR_OP_ZERO, zeros, zeros), zeros, lines);
  writeLog(lines);
  verifyBatch("one", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_ONE), FR_OP_ONE, zeros, zeros), Array.from({ length: elementCases.length }, () => oneMontHex), lines);
  writeLog(lines);
  verifyBatch("add", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_ADD), FR_OP_ADD, aHex, bHex), elementCases.map((item) => item.add_bytes_le), lines);
  writeLog(lines);
  verifyBatch("sub", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_SUB), FR_OP_SUB, aHex, bHex), elementCases.map((item) => item.sub_bytes_le), lines);
  writeLog(lines);
  verifyBatch("neg", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_NEG), FR_OP_NEG, aHex, zeros), elementCases.map((item) => item.neg_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("double", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_DOUBLE), FR_OP_DOUBLE, aHex, zeros), elementCases.map((item) => item.double_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch("mul", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_MUL), FR_OP_MUL, aHex, bHex), elementCases.map((item) => item.mul_bytes_le), lines);
  writeLog(lines);
  verifyBatch("square", await runOp(device, createKernel(device, shaderCode, elementLimbCount, FR_OP_SQUARE), FR_OP_SQUARE, aHex, zeros), elementCases.map((item) => item.square_a_bytes_le), lines);
  writeLog(lines);
  verifyBatch(
    "to_mont",
    await runOp(
      device,
      createKernel(device, shaderCode, convertLimbCount, FR_OP_TO_MONT),
      FR_OP_TO_MONT,
      vectors.convert_cases.map((item) => item.regular_bytes_le),
      Array.from({ length: vectors.convert_cases.length }, () => FR_ZERO_HEX),
    ),
    vectors.convert_cases.map((item) => item.mont_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "from_mont",
    await runOp(
      device,
      createKernel(device, shaderCode, convertLimbCount, FR_OP_FROM_MONT),
      FR_OP_FROM_MONT,
      vectors.convert_cases.map((item) => item.mont_bytes_le),
      Array.from({ length: vectors.convert_cases.length }, () => FR_ZERO_HEX),
    ),
    vectors.convert_cases.map((item) => item.regular_bytes_le),
    lines,
  );
  writeLog(lines);
  verifyBatch(
    "normalize",
    await runOp(
      device,
      createKernel(device, shaderCode, normalizeLimbCount, FR_OP_NORMALIZE),
      FR_OP_NORMALIZE,
      vectors.normalize_cases.map((item) => item.input_bytes_le),
      Array.from({ length: vectors.normalize_cases.length }, () => FR_ZERO_HEX),
    ),
    vectors.normalize_cases.map((item) => item.expected_bytes_le),
    lines,
  );
  writeLog(lines);

  lines.push("");
  lines.push("PASS: BN254 fr browser smoke succeeded");
  writeLog(lines);
  return lines;
}

async function main() {
  runButton.disabled = true;
  setPageState("running");
  setStatus("Running");
  try {
    await runBrowserSmoke();
    setPageState("pass");
    setStatus("Pass");
  } catch (error) {
    const existing = logEl.textContent ? `${logEl.textContent}\n` : "";
    logEl.textContent = `${existing}FAIL: ${error instanceof Error ? error.message : String(error)}`;
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
    "=== BN254 fr Phase 2 Browser Smoke ===",
    "",
    "Press Run to execute the BN254 fr smoke test in browser WebGPU.",
  ]);
}
