const FR_OP_TO_MONT = 11;
const FR_VECTOR_OP_MUL_FACTORS = 3;
const FR_VECTOR_OP_BIT_REVERSE_COPY = 4;
const ELEMENT_WORDS = 8;
const ELEMENT_BYTES = 32;
const UNIFORM_BYTES = 32;
const MODULUS = BigInt("0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

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

async function fetchJSON(path) {
  return JSON.parse(await fetchText(path));
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

function createStorageBuffer(device, label, size, usage) {
  return device.createBuffer({ label, size, usage });
}

function createUniformWords(a, b, c = 0) {
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = a;
  params[1] = b;
  params[2] = c;
  return params;
}

async function runArithToMont(device, arithKernel, regularWords) {
  const count = regularWords.byteLength / ELEMENT_BYTES;
  const dataBytes = regularWords.byteLength;
  const zeros = makeZeroBatch(count);
  const inputA = createStorageBuffer(device, "arith-input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "arith-input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "arith-output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "arith-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({ label: "arith-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(inputA, 0, regularWords.buffer, regularWords.byteOffset, regularWords.byteLength);
  device.queue.writeBuffer(inputB, 0, zeros);
  device.queue.writeBuffer(uniform, 0, createUniformWords(count, FR_OP_TO_MONT));

  const bindGroup = device.createBindGroup({
    layout: arithKernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(arithKernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, dataBytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  inputA.destroy();
  inputB.destroy();
  output.destroy();
  staging.destroy();
  uniform.destroy();
  return out;
}

function hexToBigInt(hex) {
  return BigInt(`0x${hex}`);
}

function modPow(base, exp, mod) {
  let result = 1n;
  let acc = base % mod;
  let power = exp;
  while (power > 0n) {
    if ((power & 1n) === 1n) {
      result = (result * acc) % mod;
    }
    acc = (acc * acc) % mod;
    power >>= 1n;
  }
  return result;
}

function bigIntToWordsLE(value) {
  const out = new Uint32Array(ELEMENT_WORDS);
  let x = value;
  for (let i = 0; i < ELEMENT_WORDS; i += 1) {
    out[i] = Number(x & 0xffffffffn);
    x >>= 32n;
  }
  return out;
}

function buildRegularStageWords(domain, inverse) {
  const logN = domain.log_n;
  const omega = hexToBigInt(inverse ? domain.omega_inv_hex : domain.omega_hex);
  const stages = [];
  for (let stage = 1; stage <= logN; stage += 1) {
    const m = 1 << (stage - 1);
    const exponentShift = BigInt(logN - stage);
    const step = modPow(omega, 1n << exponentShift, MODULUS);
    const words = new Uint32Array(m * ELEMENT_WORDS);
    let acc = 1n;
    for (let i = 0; i < m; i += 1) {
      words.set(bigIntToWordsLE(acc), i * ELEMENT_WORDS);
      acc = (acc * step) % MODULUS;
    }
    stages.push(words);
  }
  return stages;
}

async function prepareDomainMont(device, arithKernel, domain, count) {
  const inputRegular = makeRegularBatch(count, 0x9e3779b9 ^ count);
  const inputMont = await runArithToMont(device, arithKernel, inputRegular);
  const stageRegular = buildRegularStageWords(domain, false);
  const inverseStageRegular = buildRegularStageWords(domain, true);
  const stageMont = [];
  const inverseStageMont = [];
  for (const words of stageRegular) {
    stageMont.push(await runArithToMont(device, arithKernel, words));
  }
  for (const words of inverseStageRegular) {
    inverseStageMont.push(await runArithToMont(device, arithKernel, words));
  }
  const scaleRegular = new Uint32Array(ELEMENT_WORDS);
  scaleRegular.set(bigIntToWordsLE(hexToBigInt(domain.cardinality_inv_hex)));
  const scaleMont = await runArithToMont(device, arithKernel, scaleRegular);
  return {
    inputMont,
    stageMont,
    inverseStageMont,
    scaleMont: scaleMont.slice(0, ELEMENT_WORDS),
  };
}

function createStageResources(device, count, stageWordBatches) {
  const stageResources = [];
  for (const words of stageWordBatches) {
    const twiddleBuffer = createStorageBuffer(device, "ntt-twiddles", words.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const uniformBuffer = device.createBuffer({
      label: "ntt-params",
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(twiddleBuffer, 0, words.buffer, words.byteOffset, words.byteLength);
    device.queue.writeBuffer(uniformBuffer, 0, createUniformWords(count, words.length / ELEMENT_WORDS));
    stageResources.push({
      twiddleBuffer,
      uniformBuffer,
      twiddleSize: words.byteLength,
    });
  }
  return stageResources;
}

function destroyStageResources(stageResources) {
  for (const resource of stageResources) {
    resource.twiddleBuffer.destroy();
    resource.uniformBuffer.destroy();
  }
}

function dispatchVector(device, vectorKernel, inputBuffer, auxBuffer, outputBuffer, uniformBuffer, count) {
  const bindGroup = device.createBindGroup({
    layout: vectorKernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: auxBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(vectorKernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function dispatchNTTStage(device, nttKernel, inputBuffer, twiddleBuffer, outputBuffer, uniformBuffer, count) {
  const bindGroup = device.createBindGroup({
    layout: nttKernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: twiddleBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(nttKernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((count / 2) / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

async function runNTTFullPath(device, vectorKernel, nttKernel, count, inputMont, stageMont, scaleMont, inverse) {
  const totalStart = performance.now();
  const dataBytes = count * ELEMENT_BYTES;
  const zeroWords = makeZeroBatch(count);
  const inputBuffer = createStorageBuffer(device, "ntt-input", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const stateA = createStorageBuffer(device, "ntt-state-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const stateB = createStorageBuffer(device, "ntt-state-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const auxBuffer = createStorageBuffer(device, "ntt-aux", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const vectorUniform = device.createBuffer({ label: "vector-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const staging = createStorageBuffer(device, "ntt-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const uploadStart = performance.now();
  device.queue.writeBuffer(inputBuffer, 0, inputMont.buffer, inputMont.byteOffset, inputMont.byteLength);
  device.queue.writeBuffer(auxBuffer, 0, zeroWords);
  device.queue.writeBuffer(vectorUniform, 0, createUniformWords(count, FR_VECTOR_OP_BIT_REVERSE_COPY, Math.round(Math.log2(count))));
  const stageResources = createStageResources(device, count, stageMont);
  const uploadMs = performance.now() - uploadStart;

  let current = stateA;
  let next = stateB;

  const bitReverseStart = performance.now();
  dispatchVector(device, vectorKernel, inputBuffer, auxBuffer, current, vectorUniform, count);
  const bitReverseMs = performance.now() - bitReverseStart;

  const stageKernelStart = performance.now();
  for (const stage of stageResources) {
    dispatchNTTStage(device, nttKernel, current, stage.twiddleBuffer, next, stage.uniformBuffer, count);
    const tmp = current;
    current = next;
    next = tmp;
  }
  const stageKernelMs = performance.now() - stageKernelStart;

  let scaleMs = 0;
  if (inverse) {
    const scaleUploadStart = performance.now();
    const scaleBatch = new Uint32Array(count * ELEMENT_WORDS);
    for (let i = 0; i < count; i += 1) {
      scaleBatch.set(scaleMont, i * ELEMENT_WORDS);
    }
    device.queue.writeBuffer(auxBuffer, 0, scaleBatch.buffer, scaleBatch.byteOffset, scaleBatch.byteLength);
    device.queue.writeBuffer(vectorUniform, 0, createUniformWords(count, FR_VECTOR_OP_MUL_FACTORS));
    const scaleKernelStart = performance.now();
    dispatchVector(device, vectorKernel, current, auxBuffer, next, vectorUniform, count);
    current = next;
    scaleMs = (performance.now() - scaleUploadStart) + (performance.now() - scaleKernelStart);
  }

  const readbackStart = performance.now();
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(current, 0, staging, 0, dataBytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  staging.unmap();
  const readbackMs = performance.now() - readbackStart;

  inputBuffer.destroy();
  stateA.destroy();
  stateB.destroy();
  auxBuffer.destroy();
  vectorUniform.destroy();
  staging.destroy();
  destroyStageResources(stageResources);

  return {
    bitReverseMs,
    stageUploadMs: uploadMs,
    stageKernelMs,
    stageReadbackMs: readbackMs,
    stageTotalMs: uploadMs + stageKernelMs + readbackMs,
    scaleMs,
    totalMs: performance.now() - totalStart,
  };
}

async function benchmarkNTT(device, vectorKernel, nttKernel, count, inputMont, stageMont, scaleMont, inverse, iters) {
  const cold = await runNTTFullPath(device, vectorKernel, nttKernel, count, inputMont, stageMont, scaleMont, inverse);
  if (iters === 1) {
    return { cold, warm: cold };
  }
  const warm = {
    bitReverseMs: 0,
    stageUploadMs: 0,
    stageKernelMs: 0,
    stageReadbackMs: 0,
    stageTotalMs: 0,
    scaleMs: 0,
    totalMs: 0,
  };
  for (let i = 0; i < iters; i += 1) {
    const profile = await runNTTFullPath(device, vectorKernel, nttKernel, count, inputMont, stageMont, scaleMont, inverse);
    warm.bitReverseMs += profile.bitReverseMs;
    warm.stageUploadMs += profile.stageUploadMs;
    warm.stageKernelMs += profile.stageKernelMs;
    warm.stageReadbackMs += profile.stageReadbackMs;
    warm.stageTotalMs += profile.stageTotalMs;
    warm.scaleMs += profile.scaleMs;
    warm.totalMs += profile.totalMs;
  }
  warm.bitReverseMs /= iters;
  warm.stageUploadMs /= iters;
  warm.stageKernelMs /= iters;
  warm.stageReadbackMs /= iters;
  warm.stageTotalMs /= iters;
  warm.scaleMs /= iters;
  warm.totalMs /= iters;
  return { cold, warm };
}

async function runBenchmark() {
  const lines = ["=== BN254 fr NTT Browser Benchmark ===", ""];
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
    const [arithShader, vectorShader, nttShader, domainMeta] = await Promise.all([
      fetchText("/shaders/curves/bn254/fr_arith.wgsl"),
      fetchText("/shaders/curves/bn254/fr_vector.wgsl"),
      fetchText("/shaders/curves/bn254/fr_ntt.wgsl"),
      fetchJSON("/testdata/vectors/fr/bn254_ntt_domains.json"),
    ]);
    lines.push("3. Loading shaders and domain metadata... OK");

    const arithKernel = createKernel(device, "bn254-fr", arithShader, "fr_ops_main");
    const vectorKernel = createKernel(device, "bn254-fr-vector", vectorShader, "fr_vector_main");
    const nttKernel = createKernel(device, "bn254-fr-ntt", nttShader, "fr_ntt_stage_main");
    const initMs = performance.now() - initStart;
    lines.push("4. Creating pipelines... OK");
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    lines.push("");
    lines.push("size,op,init_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms");

    const domainsByLog = new Map(domainMeta.domains.map((entry) => [entry.log_n, entry]));
    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      const domain = domainsByLog.get(logSize);
      if (!domain) {
        throw new Error(`missing domain metadata for 2^${logSize}`);
      }
      const prepared = await prepareDomainMont(device, arithKernel, domain, domain.size);

      const forward = await benchmarkNTT(device, vectorKernel, nttKernel, domain.size, prepared.inputMont, prepared.stageMont, prepared.scaleMont, false, iters);
      lines.push(`${domain.size},forward_ntt,${initMs.toFixed(3)},${forward.cold.bitReverseMs.toFixed(3)},${forward.cold.stageUploadMs.toFixed(3)},${forward.cold.stageKernelMs.toFixed(3)},${forward.cold.stageReadbackMs.toFixed(3)},${forward.cold.stageTotalMs.toFixed(3)},${forward.cold.scaleMs.toFixed(3)},${forward.cold.totalMs.toFixed(3)},${(initMs + forward.cold.totalMs).toFixed(3)},${forward.warm.bitReverseMs.toFixed(3)},${forward.warm.stageTotalMs.toFixed(3)},${forward.warm.scaleMs.toFixed(3)},${forward.warm.totalMs.toFixed(3)}`);

      const inverse = await benchmarkNTT(device, vectorKernel, nttKernel, domain.size, prepared.inputMont, prepared.inverseStageMont, prepared.scaleMont, true, iters);
      lines.push(`${domain.size},inverse_ntt,${initMs.toFixed(3)},${inverse.cold.bitReverseMs.toFixed(3)},${inverse.cold.stageUploadMs.toFixed(3)},${inverse.cold.stageKernelMs.toFixed(3)},${inverse.cold.stageReadbackMs.toFixed(3)},${inverse.cold.stageTotalMs.toFixed(3)},${inverse.cold.scaleMs.toFixed(3)},${inverse.cold.totalMs.toFixed(3)},${(initMs + inverse.cold.totalMs).toFixed(3)},${inverse.warm.bitReverseMs.toFixed(3)},${inverse.warm.stageTotalMs.toFixed(3)},${inverse.warm.scaleMs.toFixed(3)},${inverse.warm.totalMs.toFixed(3)}`);
      writeLog(lines);
    }

    lines.push("");
    lines.push("PASS: BN254 fr NTT browser benchmark completed");
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
