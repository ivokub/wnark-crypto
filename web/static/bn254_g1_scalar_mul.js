const G1_OP_SCALAR_MUL_AFFINE = 8;

const FP_BYTES = 32;
const POINT_BYTES = 96;
const UNIFORM_BYTES = 32;
const ZERO_HEX = "0000000000000000000000000000000000000000000000000000000000000000";

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

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors() {
  const text = await fetchText("/testdata/vectors/g1/bn254_phase7_scalar_mul.json");
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
}

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function createStorageBuffer(device, label, size, usage) {
  return device.createBuffer({ label, size, usage });
}

function createKernel(device, shaderCode) {
  const shaderModule = device.createShaderModule({
    label: "bn254-g1-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-g1-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bn254-g1-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-g1-pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "g1_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function findOneMontZ(vectors) {
  return vectors.one_mont_z;
}

function affineToKernelPoint(point, oneMontZ) {
  const isInfinity = point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? ZERO_HEX : oneMontZ,
  };
}

function scalarToKernelPoint(scalarHex) {
  return {
    x_bytes_le: scalarHex,
    y_bytes_le: ZERO_HEX,
    z_bytes_le: ZERO_HEX,
  };
}

function packPointBatch(points) {
  const out = new Uint8Array(points.length * POINT_BYTES);
  points.forEach((point, index) => {
    const base = index * POINT_BYTES;
    out.set(hexToBytes(point.x_bytes_le), base);
    out.set(hexToBytes(point.y_bytes_le), base + FP_BYTES);
    out.set(hexToBytes(point.z_bytes_le), base + 2 * FP_BYTES);
  });
  return out;
}

function unpackPointBatch(bytes, count) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * POINT_BYTES;
    out.push({
      x_bytes_le: bytesToHex(bytes.slice(base, base + FP_BYTES)),
      y_bytes_le: bytesToHex(bytes.slice(base + FP_BYTES, base + 2 * FP_BYTES)),
      z_bytes_le: bytesToHex(bytes.slice(base + 2 * FP_BYTES, base + 3 * FP_BYTES)),
    });
  }
  return out;
}

function expectPointBatch(name, got, want) {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i].x_bytes_le !== want[i].x_bytes_le ||
      got[i].y_bytes_le !== want[i].y_bytes_le ||
      got[i].z_bytes_le !== want[i].z_bytes_le) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
}

async function runOp(device, kernel, opcode, inputA, inputB) {
  const count = inputA.length;
  const aBytes = packPointBatch(inputA);
  const bBytes = packPointBatch(inputB);
  const byteSize = count * POINT_BYTES;

  const inputABuffer = createStorageBuffer(device, "g1-input-a", byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, "g1-input-b", byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, "g1-output", byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const stagingBuffer = createStorageBuffer(device, "g1-staging", byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = device.createBuffer({
    label: "g1-params",
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputABuffer, 0, aBytes);
  device.queue.writeBuffer(inputBBuffer, 0, bBytes);
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  device.queue.writeBuffer(uniformBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    label: "g1-bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer } },
      { binding: 1, resource: { buffer: inputBBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "g1-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-pass" });
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

  return unpackPointBatch(result, count);
}

async function runSmoke() {
  const lines = ["=== BN254 G1 Phase 7 Browser Smoke ===", ""];
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

    const [shaderText, vectors] = await Promise.all([
      fetchText("/shaders/curves/bn254/g1_arith.wgsl"),
      fetchVectors(),
    ]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`cases.scalar = ${vectors.scalar_cases.length}`);
    lines.push(`cases.base = ${vectors.base_cases.length}`);

    const kernel = createKernel(device, shaderText);
    lines.push("4. Creating pipeline... OK");

    const oneMontZ = findOneMontZ(vectors);

    const scalarBaseInputs = vectors.scalar_cases.map((item) => affineToKernelPoint(item.base_affine, oneMontZ));
    const scalarInputs = vectors.scalar_cases.map((item) => scalarToKernelPoint(item.scalar_bytes_le));
    const scalarWant = vectors.scalar_cases.map((item) => item.scalar_mul_affine);

    expectPointBatch("scalar_mul_affine", await runOp(device, kernel, G1_OP_SCALAR_MUL_AFFINE, scalarBaseInputs, scalarInputs), scalarWant);
    lines.push("scalar_mul_affine: OK");

    const generatorKernel = affineToKernelPoint(vectors.generator_affine, oneMontZ);
    const baseInputs = vectors.base_cases.map(() => generatorKernel);
    const baseScalarInputs = vectors.base_cases.map((item) => scalarToKernelPoint(item.scalar_bytes_le));
    const baseWant = vectors.base_cases.map((item) => item.scalar_mul_base_affine);

    expectPointBatch("scalar_mul_base_affine", await runOp(device, kernel, G1_OP_SCALAR_MUL_AFFINE, baseInputs, baseScalarInputs), baseWant);
    lines.push("scalar_mul_base_affine: OK");

    lines.push("");
    lines.push("PASS: BN254 G1 Phase 7 browser smoke succeeded");
    writeLog(lines);
    setStatus("PASS");
    setPageState("pass");
  } catch (error) {
    lines.push("");
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("FAIL");
    setPageState("fail");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void runSmoke();
});
