const FR_OP_TO_MONT = 11;

const FR_VECTOR_OP_ADD = 1;
const FR_VECTOR_OP_SUB = 2;
const FR_VECTOR_OP_MUL_FACTORS = 3;
const FR_VECTOR_OP_BIT_REVERSE_COPY = 4;

const ELEMENT_WORDS = 8;
const ELEMENT_BYTES = 32;
const UNIFORM_BYTES = 32;

const minLogEl = document.getElementById("min-log");
const maxLogEl = document.getElementById("max-log");
const itersEl = document.getElementById("iters");
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

function createKernel(device, label, shaderCode, entryPoint) {
  const shaderModule = device.createShaderModule({ label: `${label}-shader`, code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${label}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${label}-pipeline`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

function makeRegularBatch(count, seed) {
  const words = new Uint32Array(count * ELEMENT_WORDS);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    words[i * ELEMENT_WORDS + 0] = state >>> 0;
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    words[i * ELEMENT_WORDS + 1] = state >>> 0;
  }
  return words;
}

function makeZeroBatch(count) {
  return new Uint32Array(count * ELEMENT_WORDS);
}

async function runFullPathProfiled(device, kernel, inputA, inputB, opcode, logCount) {
  const count = inputA.byteLength / ELEMENT_BYTES;
  const dataBytes = inputA.byteLength;
  const totalStart = performance.now();
  const inputABuffer = device.createBuffer({
    label: "input-a",
    size: dataBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const inputBBuffer = device.createBuffer({
    label: "input-b",
    size: dataBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    label: "output",
    size: dataBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuffer = device.createBuffer({
    label: "staging",
    size: dataBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const uniformBuffer = device.createBuffer({
    label: "params",
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uploadStart = performance.now();
  device.queue.writeBuffer(inputABuffer, 0, inputA.buffer, inputA.byteOffset, inputA.byteLength);
  device.queue.writeBuffer(inputBBuffer, 0, inputB.buffer, inputB.byteOffset, inputB.byteLength);
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  params[2] = logCount;
  device.queue.writeBuffer(uniformBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    label: "bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer } },
      { binding: 1, resource: { buffer: inputBBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });
  const uploadMs = performance.now() - uploadStart;

  const kernelStart = performance.now();
  const encoder = device.createCommandEncoder({ label: "encoder" });
  const pass = encoder.beginComputePass({ label: "pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, dataBytes);
  device.queue.submit([encoder.finish()]);
  const kernelMs = performance.now() - kernelStart;

  const readbackStart = performance.now();
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();
  const readbackMs = performance.now() - readbackStart;

  inputABuffer.destroy();
  inputBBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();

  return {
    out,
    profile: {
      uploadMs,
      kernelMs,
      readbackMs,
      totalMs: performance.now() - totalStart,
    },
  };
}

async function toMontBatch(device, arithKernel, regularWords) {
  const zeros = makeZeroBatch(regularWords.byteLength / ELEMENT_BYTES);
  return (await runFullPathProfiled(device, arithKernel, regularWords, zeros, FR_OP_TO_MONT, 0)).out;
}

async function benchOp(device, kernel, inputA, inputB, opcode, logCount, iters) {
  const cold = await runFullPathProfiled(device, kernel, inputA, inputB, opcode, logCount);
  if (iters === 1) {
    return { cold: cold.profile, warm: cold.profile };
  }
  let uploadMs = 0;
  let kernelMs = 0;
  let readbackMs = 0;
  let totalMs = 0;
  for (let i = 0; i < iters; i += 1) {
    const warm = await runFullPathProfiled(device, kernel, inputA, inputB, opcode, logCount);
    uploadMs += warm.profile.uploadMs;
    kernelMs += warm.profile.kernelMs;
    readbackMs += warm.profile.readbackMs;
    totalMs += warm.profile.totalMs;
  }
  return {
    cold: cold.profile,
    warm: {
      uploadMs: uploadMs / iters,
      kernelMs: kernelMs / iters,
      readbackMs: readbackMs / iters,
      totalMs: totalMs / iters,
    },
  };
}

async function runBenchmark() {
  const lines = ["=== BN254 fr Vector Browser Benchmark ===", ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    const minLog = Number.parseInt(minLogEl.value, 10);
    const maxLog = Number.parseInt(maxLogEl.value, 10);
    const iters = Number.parseInt(itersEl.value, 10);
    if (!Number.isInteger(minLog) || !Number.isInteger(maxLog) || !Number.isInteger(iters) || minLog < 1 || maxLog < minLog || iters < 1) {
      throw new Error("invalid benchmark controls");
    }
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }

    const initStart = performance.now();
    lines.push("1. Requesting adapter... OK");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    await appendAdapterDiagnostics(adapter, lines);
    lines.push("2. Requesting device... OK");
    const device = await adapter.requestDevice();

    const [arithShader, vectorShader] = await Promise.all([
      fetchText("/shaders/curves/bn254/fr_arith.wgsl"),
      fetchText("/shaders/curves/bn254/fr_vector.wgsl"),
    ]);
    lines.push("3. Loading shaders... OK");

    const arithKernel = createKernel(device, "bn254-fr", arithShader, "fr_ops_main");
    const vectorKernel = createKernel(device, "bn254-fr-vector", vectorShader, "fr_vector_main");
    const initElapsed = performance.now() - initStart;
    lines.push("4. Creating pipelines... OK");
    lines.push(`init_ms = ${initElapsed.toFixed(3)}`);
    lines.push("");
    lines.push("size,op,init_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_total_ms,cold_with_init_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_total_ms");

    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      const size = 1 << logSize;
      const leftRegular = makeRegularBatch(size, 0x12345678 ^ size);
      const rightRegular = makeRegularBatch(size, 0x9e3779b9 ^ size);
      const zeros = makeZeroBatch(size);

      const leftMont = await toMontBatch(device, arithKernel, leftRegular);
      const rightMont = await toMontBatch(device, arithKernel, rightRegular);

      const addBench = await benchOp(device, vectorKernel, leftMont, rightMont, FR_VECTOR_OP_ADD, 0, iters);
      lines.push(`${size},add,${initElapsed.toFixed(3)},${addBench.cold.uploadMs.toFixed(3)},${addBench.cold.kernelMs.toFixed(3)},${addBench.cold.readbackMs.toFixed(3)},${addBench.cold.totalMs.toFixed(3)},${(initElapsed + addBench.cold.totalMs).toFixed(3)},${addBench.warm.uploadMs.toFixed(3)},${addBench.warm.kernelMs.toFixed(3)},${addBench.warm.readbackMs.toFixed(3)},${addBench.warm.totalMs.toFixed(3)}`);

      const subBench = await benchOp(device, vectorKernel, leftMont, rightMont, FR_VECTOR_OP_SUB, 0, iters);
      lines.push(`${size},sub,${initElapsed.toFixed(3)},${subBench.cold.uploadMs.toFixed(3)},${subBench.cold.kernelMs.toFixed(3)},${subBench.cold.readbackMs.toFixed(3)},${subBench.cold.totalMs.toFixed(3)},${(initElapsed + subBench.cold.totalMs).toFixed(3)},${subBench.warm.uploadMs.toFixed(3)},${subBench.warm.kernelMs.toFixed(3)},${subBench.warm.readbackMs.toFixed(3)},${subBench.warm.totalMs.toFixed(3)}`);

      const mulBench = await benchOp(device, vectorKernel, leftMont, rightMont, FR_VECTOR_OP_MUL_FACTORS, 0, iters);
      lines.push(`${size},mul,${initElapsed.toFixed(3)},${mulBench.cold.uploadMs.toFixed(3)},${mulBench.cold.kernelMs.toFixed(3)},${mulBench.cold.readbackMs.toFixed(3)},${mulBench.cold.totalMs.toFixed(3)},${(initElapsed + mulBench.cold.totalMs).toFixed(3)},${mulBench.warm.uploadMs.toFixed(3)},${mulBench.warm.kernelMs.toFixed(3)},${mulBench.warm.readbackMs.toFixed(3)},${mulBench.warm.totalMs.toFixed(3)}`);

      const bitReverseBench = await benchOp(device, vectorKernel, leftMont, zeros, FR_VECTOR_OP_BIT_REVERSE_COPY, logSize, iters);
      lines.push(`${size},bit_reverse,${initElapsed.toFixed(3)},${bitReverseBench.cold.uploadMs.toFixed(3)},${bitReverseBench.cold.kernelMs.toFixed(3)},${bitReverseBench.cold.readbackMs.toFixed(3)},${bitReverseBench.cold.totalMs.toFixed(3)},${(initElapsed + bitReverseBench.cold.totalMs).toFixed(3)},${bitReverseBench.warm.uploadMs.toFixed(3)},${bitReverseBench.warm.kernelMs.toFixed(3)},${bitReverseBench.warm.readbackMs.toFixed(3)},${bitReverseBench.warm.totalMs.toFixed(3)}`);
    }

    lines.push("");
    lines.push("PASS: BN254 fr vector browser benchmark completed");
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
  void runBenchmark();
});
