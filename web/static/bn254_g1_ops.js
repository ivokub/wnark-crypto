const G1_OP_COPY = 0;
const G1_OP_JAC_INFINITY = 1;
const G1_OP_AFFINE_TO_JAC = 2;
const G1_OP_NEG_JAC = 3;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;

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
  const text = await fetchText("/testdata/vectors/g1/bn254_phase6_g1_ops.json");
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
  for (const pointCase of vectors.point_cases) {
    if (pointCase.p_jacobian.z_bytes_le !== ZERO_HEX) {
      return pointCase.p_jacobian.z_bytes_le;
    }
  }
  throw new Error("missing non-zero affine flag in vectors");
}

function affineToKernelPoint(point, oneMontZ) {
  const isInfinity = point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? ZERO_HEX : oneMontZ,
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
  const lines = ["=== BN254 G1 Phase 6 Browser Smoke ===", ""];
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
    lines.push(`cases.g1 = ${vectors.point_cases.length}`);

    const kernel = createKernel(device, shaderText);
    lines.push("4. Creating pipeline... OK");

    const oneMontZ = findOneMontZ(vectors);
    const pAffineInputs = vectors.point_cases.map((pointCase) => affineToKernelPoint(pointCase.p_affine, oneMontZ));
    const qAffineInputs = vectors.point_cases.map((pointCase) => affineToKernelPoint(pointCase.q_affine, oneMontZ));
    const pJac = vectors.point_cases.map((pointCase) => pointCase.p_jacobian);
    const negWant = vectors.point_cases.map((pointCase) => pointCase.neg_p_jacobian);
    const doubleWant = vectors.point_cases.map((pointCase) => pointCase.double_p_jacobian);
    const addWant = vectors.point_cases.map((pointCase) => pointCase.add_mixed_p_plus_q_jacobian);
    const affineWant = vectors.point_cases.map((pointCase) => pointCase.p_affine_output);
    const affineAddWant = vectors.point_cases.map((pointCase) => pointCase.affine_add_p_plus_q);
    const infinityWant = vectors.point_cases.map(() => ({ x_bytes_le: oneMontZ, y_bytes_le: oneMontZ, z_bytes_le: ZERO_HEX }));
    const zeros = vectors.point_cases.map(() => ({ x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX }));

    expectPointBatch("copy", await runOp(device, kernel, G1_OP_COPY, pJac, zeros), pJac);
    lines.push("copy: OK");

    expectPointBatch("jac_infinity", await runOp(device, kernel, G1_OP_JAC_INFINITY, zeros, zeros), infinityWant);
    lines.push("jac_infinity: OK");

    expectPointBatch("affine_to_jac", await runOp(device, kernel, G1_OP_AFFINE_TO_JAC, pAffineInputs, zeros), pJac);
    lines.push("affine_to_jac: OK");

    expectPointBatch("neg_jac", await runOp(device, kernel, G1_OP_NEG_JAC, pJac, zeros), negWant);
    lines.push("neg_jac: OK");

    expectPointBatch("jac_to_affine", await runOp(device, kernel, G1_OP_JAC_TO_AFFINE, pJac, zeros), affineWant);
    lines.push("jac_to_affine: OK");

    expectPointBatch("double_jac", await runOp(device, kernel, G1_OP_DOUBLE_JAC, pJac, zeros), doubleWant);
    lines.push("double_jac: OK");

    expectPointBatch("add_mixed", await runOp(device, kernel, G1_OP_ADD_MIXED, pJac, qAffineInputs), addWant);
    lines.push("add_mixed: OK");

    expectPointBatch("affine_add", await runOp(device, kernel, G1_OP_AFFINE_ADD, pAffineInputs, qAffineInputs), affineAddWant);
    lines.push("affine_add: OK");

    lines.push("");
    lines.push("PASS: BN254 G1 Phase 6 browser smoke succeeded");
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
