const FR_OP_ADD = 3;
const FR_OP_SUB = 4;
const FR_OP_MUL = 9;
const FR_OP_TO_MONT = 11;
const FR_OP_FROM_MONT = 12;

const FR_VECTOR_OP_MUL_FACTORS = 1;
const FR_VECTOR_OP_BIT_REVERSE_COPY = 2;

const ELEMENT_BYTES = 32;
const UNIFORM_BYTES = 32;

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
  const text = await fetchText("/testdata/vectors/fr/bn254_phase4_vector_ops.json");
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

function packHexBatch(hexValues) {
  const out = new Uint8Array(hexValues.length * ELEMENT_BYTES);
  hexValues.forEach((hex, index) => {
    out.set(hexToBytes(hex), index * ELEMENT_BYTES);
  });
  return out;
}

function createStorageBuffer(device, label, size, usage) {
  return device.createBuffer({ label, size, usage });
}

function createArithKernel(device, shaderCode) {
  const shaderModule = device.createShaderModule({
    label: "bn254-fr-shader",
    code: shaderCode,
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
    compute: { module: shaderModule, entryPoint: "fr_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function createVectorKernel(device, shaderCode) {
  const shaderModule = device.createShaderModule({
    label: "bn254-fr-vector-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-fr-vector-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bn254-fr-vector-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-fr-vector-pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "fr_vector_main" },
  });
  return { pipeline, bindGroupLayout };
}

async function runKernel(device, kernel, aHex, bHex, opcode, logCount) {
  const count = aHex.length;
  const dataBytes = count * ELEMENT_BYTES;
  const inputA = createStorageBuffer(device, "input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({
    label: "params",
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputA, 0, packHexBatch(aHex));
  device.queue.writeBuffer(inputB, 0, packHexBatch(bHex));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  params[2] = logCount;
  device.queue.writeBuffer(uniform, 0, params);

  const bindGroup = device.createBindGroup({
    label: "bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "encoder" });
  const pass = encoder.beginComputePass({ label: "pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, dataBytes);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const view = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();

  inputA.destroy();
  inputB.destroy();
  output.destroy();
  staging.destroy();
  uniform.destroy();

  const out = [];
  for (let i = 0; i < count; i += 1) {
    out.push(bytesToHex(view.slice(i * ELEMENT_BYTES, (i + 1) * ELEMENT_BYTES)));
  }
  return out;
}

function expectBatch(name, got, want) {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i] !== want[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${got[i]} want=${want[i]}`);
    }
  }
}

async function runSmoke() {
  const lines = ["=== BN254 fr Phase 4 Browser Smoke ===", ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }
    lines.push("1. Requesting adapter... OK");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    await appendAdapterDiagnostics(adapter, lines);
    lines.push("2. Requesting device... OK");
    const device = await adapter.requestDevice();

    const [arithShader, vectorShader, vectors] = await Promise.all([
      fetchText("/shaders/curves/bn254/fr_arith.wgsl"),
      fetchText("/shaders/curves/bn254/fr_vector.wgsl"),
      fetchVectors(),
    ]);
    lines.push("3. Loading shaders and vectors... OK");
    lines.push(`cases.vector = ${vectors.vector_cases.length}`);

    const arithKernel = createArithKernel(device, arithShader);
    const vectorKernel = createVectorKernel(device, vectorShader);
    lines.push("4. Creating pipelines... OK");

    for (const vectorCase of vectors.vector_cases) {
      const zeros = vectorCase.mont_inputs_le.map(() => "0000000000000000000000000000000000000000000000000000000000000000");
      expectBatch(`${vectorCase.name}:add`, await runKernel(device, arithKernel, vectorCase.mont_inputs_le, vectorCase.mont_factors_le, FR_OP_ADD, 0), vectorCase.add_expected_le);
      expectBatch(`${vectorCase.name}:sub`, await runKernel(device, arithKernel, vectorCase.mont_inputs_le, vectorCase.mont_factors_le, FR_OP_SUB, 0), vectorCase.sub_expected_le);
      expectBatch(`${vectorCase.name}:mul`, await runKernel(device, arithKernel, vectorCase.mont_inputs_le, vectorCase.mont_factors_le, FR_OP_MUL, 0), vectorCase.mul_expected_le);
      expectBatch(`${vectorCase.name}:to_mont`, await runKernel(device, arithKernel, vectorCase.regular_inputs_le, zeros, FR_OP_TO_MONT, 0), vectorCase.to_mont_expected_le);
      expectBatch(`${vectorCase.name}:from_mont`, await runKernel(device, arithKernel, vectorCase.mont_inputs_le, zeros, FR_OP_FROM_MONT, 0), vectorCase.from_mont_expected_le);
      expectBatch(`${vectorCase.name}:mul_factors`, await runKernel(device, vectorKernel, vectorCase.mont_inputs_le, vectorCase.mont_factors_le, FR_VECTOR_OP_MUL_FACTORS, 0), vectorCase.mul_expected_le);
      const logCount = Math.round(Math.log2(vectorCase.mont_inputs_le.length));
      expectBatch(`${vectorCase.name}:bit_reverse_copy`, await runKernel(device, vectorKernel, vectorCase.mont_inputs_le, zeros, FR_VECTOR_OP_BIT_REVERSE_COPY, logCount), vectorCase.bit_reverse_expected_le);
    }

    lines.push("add: OK");
    lines.push("sub: OK");
    lines.push("mul: OK");
    lines.push("to_mont: OK");
    lines.push("from_mont: OK");
    lines.push("mul_factors: OK");
    lines.push("bit_reverse_copy: OK");
    lines.push("");
    lines.push("PASS: BN254 fr Phase 4 browser smoke succeeded");
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
  void runSmoke();
});
