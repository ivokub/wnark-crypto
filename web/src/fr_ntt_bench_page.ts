export {};

type DomainMetadata = {
  log_n: number;
  size: number;
  omega_hex: string;
  omega_inv_hex: string;
  cardinality_inv_hex: string;
};

type DomainMetadataFile = {
  domains: DomainMetadata[];
};

type NTTConfig = {
  curve: string;
  title: string;
  supported: boolean;
  unsupportedReason?: string;
  arithShaderPath?: string;
  vectorShaderPath?: string;
  nttShaderPath?: string;
  domainPath?: string;
  modulus?: bigint;
};

const FR_OP_TO_MONT = 11;
const FR_VECTOR_OP_MUL_FACTORS = 3;
const FR_VECTOR_OP_BIT_REVERSE_COPY = 4;
const ELEMENT_WORDS = 8;
const ELEMENT_BYTES = 32;
const UNIFORM_BYTES = 32;

type Kernel = {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
};

type StageResource = {
  twiddleBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
};

type NTTProfile = {
  bitReverseMs: number;
  stageUploadMs: number;
  stageKernelMs: number;
  stageReadbackMs: number;
  stageTotalMs: number;
  scaleMs: number;
  totalMs: number;
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

const CONFIGS: Record<string, NTTConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 fr NTT Browser Benchmark",
    supported: true,
    arithShaderPath: "/shaders/curves/bn254/fr_arith.wgsl",
    vectorShaderPath: "/shaders/curves/bn254/fr_vector.wgsl",
    nttShaderPath: "/shaders/curves/bn254/fr_ntt.wgsl",
    domainPath: "/testdata/vectors/fr/bn254_ntt_domains.json",
    modulus: BigInt("0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"),
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 fr NTT Browser Benchmark",
    supported: true,
    arithShaderPath: "/shaders/curves/bls12_381/fr_arith.wgsl",
    vectorShaderPath: "/shaders/curves/bls12_381/fr_vector.wgsl",
    nttShaderPath: "/shaders/curves/bls12_381/fr_ntt.wgsl",
    domainPath: "/testdata/vectors/fr/bls12_381_ntt_domains.json",
    modulus: BigInt("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"),
  },
};

const minLogEl = document.getElementById("min-log") as HTMLInputElement | null;
const maxLogEl = document.getElementById("max-log") as HTMLInputElement | null;
const itersEl = document.getElementById("iters") as HTMLInputElement | null;
const runButton = document.getElementById("run") as HTMLButtonElement | null;
const statusEl = document.getElementById("status") as HTMLElement | null;
const logEl = document.getElementById("log") as HTMLElement | null;

function getConfig(): NTTConfig {
  const curve = new URLSearchParams(window.location.search).get("curve") ?? "bn254";
  const config = CONFIGS[curve];
  if (!config) {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return config;
}

function mustElement<T>(value: T | null, name: string): T {
  if (value === null) {
    throw new Error(`missing element: ${name}`);
  }
  return value;
}

function setStatus(text: string): void {
  mustElement(statusEl, "status").textContent = text;
}

function setPageState(state: string): void {
  document.body.dataset.status = state;
}

function writeLog(lines: string[]): void {
  mustElement(logEl, "log").textContent = lines.join("\n");
}

async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchJSON<T>(path: string): Promise<T> {
  return JSON.parse(await fetchText(path)) as T;
}

async function getAdapterInfo(adapter: GPUAdapter): Promise<GPUAdapterInfo | null> {
  const adapterWithInfo = adapter as GPUAdapter & { info?: GPUAdapterInfo; requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
  if (adapterWithInfo.info) {
    return adapterWithInfo.info;
  }
  if (typeof adapterWithInfo.requestAdapterInfo === "function") {
    try {
      return await adapterWithInfo.requestAdapterInfo();
    } catch {
      return null;
    }
  }
  return null;
}

async function appendAdapterDiagnostics(adapter: GPUAdapter, lines: string[]): Promise<void> {
  const adapterWithFallback = adapter as GPUAdapter & { isFallbackAdapter?: boolean };
  if ("isFallbackAdapter" in adapterWithFallback) {
    lines.push(`adapter.isFallbackAdapter = ${String(adapterWithFallback.isFallbackAdapter)}`);
  }
  const info = await getAdapterInfo(adapter);
  if (!info) {
    lines.push("adapter.info = unavailable");
    return;
  }
  if (info.vendor) lines.push(`adapter.vendor = ${info.vendor}`);
  if (info.architecture) lines.push(`adapter.architecture = ${info.architecture}`);
}

function createKernel(device: GPUDevice, label: string, shaderCode: string, entryPoint: string): Kernel {
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
  const pipelineLayout = device.createPipelineLayout({ label: `${label}-pl`, bindGroupLayouts: [bindGroupLayout] });
  const pipeline = device.createComputePipeline({ label: `${label}-pipeline`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
  return { pipeline, bindGroupLayout };
}

function makeRegularBatch(count: number, seed: number): Uint32Array {
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

function makeZeroBatch(count: number): Uint32Array {
  return new Uint32Array(count * ELEMENT_WORDS);
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: number): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function createUniformWords(a: number, b: number, c = 0): Uint32Array {
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = a;
  params[1] = b;
  params[2] = c;
  return params;
}

async function runArithToMont(device: GPUDevice, arithKernel: Kernel, regularWords: Uint32Array): Promise<Uint32Array> {
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
  inputA.destroy(); inputB.destroy(); output.destroy(); staging.destroy(); uniform.destroy();
  return out;
}

function hexToBigInt(hex: string): bigint {
  return BigInt(`0x${hex}`);
}

function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  let acc = base % mod;
  let power = exp;
  while (power > 0n) {
    if ((power & 1n) === 1n) result = (result * acc) % mod;
    acc = (acc * acc) % mod;
    power >>= 1n;
  }
  return result;
}

function bigIntToWordsLE(value: bigint): Uint32Array {
  const out = new Uint32Array(ELEMENT_WORDS);
  let x = value;
  for (let i = 0; i < ELEMENT_WORDS; i += 1) {
    out[i] = Number(x & 0xffffffffn);
    x >>= 32n;
  }
  return out;
}

function buildRegularStageWords(domain: DomainMetadata, inverse: boolean, modulus: bigint): Uint32Array[] {
  const logN = domain.log_n;
  const omega = hexToBigInt(inverse ? domain.omega_inv_hex : domain.omega_hex);
  const stages: Uint32Array[] = [];
  for (let stage = 1; stage <= logN; stage += 1) {
    const m = 1 << (stage - 1);
    const exponentShift = BigInt(logN - stage);
    const step = modPow(omega, 1n << exponentShift, modulus);
    const words = new Uint32Array(m * ELEMENT_WORDS);
    let acc = 1n;
    for (let i = 0; i < m; i += 1) {
      words.set(bigIntToWordsLE(acc), i * ELEMENT_WORDS);
      acc = (acc * step) % modulus;
    }
    stages.push(words);
  }
  return stages;
}

async function prepareDomainMont(device: GPUDevice, arithKernel: Kernel, domain: DomainMetadata, count: number, modulus: bigint): Promise<{
  inputMont: Uint32Array;
  stageMont: Uint32Array[];
  inverseStageMont: Uint32Array[];
  scaleMont: Uint32Array;
}> {
  const inputRegular = makeRegularBatch(count, 0x9e3779b9 ^ count);
  const inputMont = await runArithToMont(device, arithKernel, inputRegular);
  const stageRegular = buildRegularStageWords(domain, false, modulus);
  const inverseStageRegular = buildRegularStageWords(domain, true, modulus);
  const stageMont: Uint32Array[] = [];
  const inverseStageMont: Uint32Array[] = [];
  for (const stage of stageRegular) stageMont.push(await runArithToMont(device, arithKernel, stage));
  for (const stage of inverseStageRegular) inverseStageMont.push(await runArithToMont(device, arithKernel, stage));
  const scaleRegular = new Uint32Array(ELEMENT_WORDS);
  scaleRegular.set(bigIntToWordsLE(hexToBigInt(domain.cardinality_inv_hex)));
  const scaleMont = await runArithToMont(device, arithKernel, scaleRegular);
  return { inputMont, stageMont, inverseStageMont, scaleMont };
}

async function runBitReverseProfiled(device: GPUDevice, vectorKernel: Kernel, inputWords: Uint32Array): Promise<{ output: Uint32Array; ms: number }> {
  const count = inputWords.byteLength / ELEMENT_BYTES;
  const dataBytes = inputWords.byteLength;
  const zeros = makeZeroBatch(count);
  const inputA = createStorageBuffer(device, "bitrev-input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "bitrev-input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "bitrev-output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "bitrev-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({ label: "bitrev-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inputA, 0, inputWords.buffer, inputWords.byteOffset, inputWords.byteLength);
  device.queue.writeBuffer(inputB, 0, zeros);
  device.queue.writeBuffer(uniform, 0, createUniformWords(count, FR_VECTOR_OP_BIT_REVERSE_COPY, Math.round(Math.log2(count))));
  const bindGroup = device.createBindGroup({
    layout: vectorKernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });
  const start = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(vectorKernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, dataBytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  const ms = performance.now() - start;
  inputA.destroy(); inputB.destroy(); output.destroy(); staging.destroy(); uniform.destroy();
  return { output: out, ms };
}

async function runMulFactorsProfiled(device: GPUDevice, vectorKernel: Kernel, inputWords: Uint32Array, factorWords: Uint32Array): Promise<{ output: Uint32Array; ms: number }> {
  const count = inputWords.byteLength / ELEMENT_BYTES;
  const dataBytes = inputWords.byteLength;
  const repeated = new Uint32Array(count * ELEMENT_WORDS);
  for (let i = 0; i < count; i += 1) repeated.set(factorWords, i * ELEMENT_WORDS);
  const inputA = createStorageBuffer(device, "mul-input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "mul-input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "mul-output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "mul-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({ label: "mul-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inputA, 0, inputWords.buffer, inputWords.byteOffset, inputWords.byteLength);
  device.queue.writeBuffer(inputB, 0, repeated.buffer, repeated.byteOffset, repeated.byteLength);
  device.queue.writeBuffer(uniform, 0, createUniformWords(count, FR_VECTOR_OP_MUL_FACTORS));
  const bindGroup = device.createBindGroup({
    layout: vectorKernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });
  const start = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(vectorKernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, dataBytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  const ms = performance.now() - start;
  inputA.destroy(); inputB.destroy(); output.destroy(); staging.destroy(); uniform.destroy();
  return { output: out, ms };
}

function createStageResource(device: GPUDevice, stageWords: Uint32Array, stageIndex: number): StageResource {
  const twiddleBuffer = createStorageBuffer(device, `ntt-stage-twiddles-${stageIndex}`, stageWords.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(twiddleBuffer, 0, stageWords.buffer, stageWords.byteOffset, stageWords.byteLength);
  const uniformBuffer = device.createBuffer({ label: `ntt-stage-params-${stageIndex}`, size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[1] = 1 << stageIndex;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);
  return { twiddleBuffer, uniformBuffer };
}

async function benchmarkNTT(
  device: GPUDevice,
  vectorKernel: Kernel,
  nttKernel: Kernel,
  size: number,
  inputMont: Uint32Array,
  stageMont: readonly Uint32Array[],
  scaleMont: Uint32Array,
  inverse: boolean,
  iters: number,
): Promise<{ cold: NTTProfile; warm: NTTProfile }> {
  async function runOnce(): Promise<NTTProfile> {
    let total = 0;
    const bitReverse = await runBitReverseProfiled(device, vectorKernel, inputMont);
    total += bitReverse.ms;
    let state = bitReverse.output;
    let stageUploadMs = 0;
    let stageKernelMs = 0;
    let stageReadbackMs = 0;
    for (let i = 0; i < stageMont.length; i += 1) {
      const stageRes = createStageResource(device, stageMont[i], i);
      stageUploadMs += 0;
      const dataBytes = state.byteLength;
      const inputA = createStorageBuffer(device, "ntt-stage-input", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const output = createStorageBuffer(device, "ntt-stage-output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const staging = createStorageBuffer(device, "ntt-stage-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(inputA, 0, state.buffer, state.byteOffset, state.byteLength);
      const params = new Uint32Array(UNIFORM_BYTES / 4);
      params[0] = size;
      params[1] = 1 << i;
      params[2] = inverse ? 1 : 0;
      device.queue.writeBuffer(stageRes.uniformBuffer, 0, params.buffer);
      const bindGroup = device.createBindGroup({
        layout: nttKernel.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: inputA } },
          { binding: 1, resource: { buffer: stageRes.twiddleBuffer } },
          { binding: 2, resource: { buffer: output } },
          { binding: 3, resource: { buffer: stageRes.uniformBuffer } },
        ],
      });
      const kernelStart = performance.now();
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(nttKernel.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(size / 64));
      pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, dataBytes);
      device.queue.submit([encoder.finish()]);
      stageKernelMs += performance.now() - kernelStart;
      const readStart = performance.now();
      await staging.mapAsync(GPUMapMode.READ);
      state = new Uint32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      stageReadbackMs += performance.now() - readStart;
      inputA.destroy(); output.destroy(); staging.destroy(); stageRes.twiddleBuffer.destroy(); stageRes.uniformBuffer.destroy();
    }
    const stageTotalMs = stageUploadMs + stageKernelMs + stageReadbackMs;
    total += stageTotalMs;
    let scaleMs = 0;
    if (inverse) {
      const scaled = await runMulFactorsProfiled(device, vectorKernel, state, scaleMont);
      scaleMs = scaled.ms;
      total += scaleMs;
    }
    return { bitReverseMs: bitReverse.ms, stageUploadMs, stageKernelMs, stageReadbackMs, stageTotalMs, scaleMs, totalMs: total };
  }

  const cold = await runOnce();
  let warmAcc: NTTProfile = { bitReverseMs: 0, stageUploadMs: 0, stageKernelMs: 0, stageReadbackMs: 0, stageTotalMs: 0, scaleMs: 0, totalMs: 0 };
  for (let i = 0; i < iters; i += 1) {
    const run = await runOnce();
    warmAcc = {
      bitReverseMs: warmAcc.bitReverseMs + run.bitReverseMs,
      stageUploadMs: warmAcc.stageUploadMs + run.stageUploadMs,
      stageKernelMs: warmAcc.stageKernelMs + run.stageKernelMs,
      stageReadbackMs: warmAcc.stageReadbackMs + run.stageReadbackMs,
      stageTotalMs: warmAcc.stageTotalMs + run.stageTotalMs,
      scaleMs: warmAcc.scaleMs + run.scaleMs,
      totalMs: warmAcc.totalMs + run.totalMs,
    };
  }
  const denom = Math.max(1, iters);
  const warm: NTTProfile = {
    bitReverseMs: warmAcc.bitReverseMs / denom,
    stageUploadMs: warmAcc.stageUploadMs / denom,
    stageKernelMs: warmAcc.stageKernelMs / denom,
    stageReadbackMs: warmAcc.stageReadbackMs / denom,
    stageTotalMs: warmAcc.stageTotalMs / denom,
    scaleMs: warmAcc.scaleMs / denom,
    totalMs: warmAcc.totalMs / denom,
  };
  return { cold, warm };
}

async function runBenchmark(): Promise<void> {
  const config = getConfig();
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  mustElement(runButton, "run").disabled = true;

  try {
    if (!config.supported) {
      throw new Error(config.unsupportedReason ?? `NTT is not implemented for ${config.curve}`);
    }
    const minLog = Number.parseInt(mustElement(minLogEl, "min-log").value, 10);
    const maxLog = Number.parseInt(mustElement(maxLogEl, "max-log").value, 10);
    const iters = Number.parseInt(mustElement(itersEl, "iters").value, 10);
    if (!Number.isInteger(minLog) || !Number.isInteger(maxLog) || !Number.isInteger(iters) || minLog < 1 || maxLog < minLog || iters < 1) {
      throw new Error("invalid benchmark controls");
    }
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }

    const initStart = performance.now();
    lines.push("1. Requesting adapter... OK");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("requestAdapter returned null");
    await appendAdapterDiagnostics(adapter, lines);
    lines.push("2. Requesting device... OK");
    const device = await adapter.requestDevice();
    const [arithShader, vectorShader, nttShader, domainMeta] = await Promise.all([
      fetchText(config.arithShaderPath!),
      fetchText(config.vectorShaderPath!),
      fetchText(config.nttShaderPath!),
      fetchJSON<DomainMetadataFile>(config.domainPath!),
    ]);
    lines.push("3. Loading shaders and domain metadata... OK");
    const arithKernel = createKernel(device, `${config.curve}-fr`, arithShader, "fr_ops_main");
    const vectorKernel = createKernel(device, `${config.curve}-fr-vector`, vectorShader, "fr_vector_main");
    const nttKernel = createKernel(device, `${config.curve}-fr-ntt`, nttShader, "fr_ntt_stage_main");
    const initMs = performance.now() - initStart;
    lines.push("4. Creating pipelines... OK");
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    lines.push("");
    lines.push("size,op,init_ms,cold_bit_reverse_ms,cold_stage_upload_ms,cold_stage_kernel_ms,cold_stage_readback_ms,cold_stage_total_ms,cold_scale_ms,cold_total_ms,cold_with_init_ms,warm_bit_reverse_ms,warm_stage_total_ms,warm_scale_ms,warm_total_ms");

    const domainsByLog = new Map<number, DomainMetadata>(domainMeta.domains.map((entry) => [entry.log_n, entry]));
    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      const domain = domainsByLog.get(logSize);
      if (!domain) throw new Error(`missing domain metadata for 2^${logSize}`);
      const prepared = await prepareDomainMont(device, arithKernel, domain, domain.size, config.modulus!);
      const forward = await benchmarkNTT(device, vectorKernel, nttKernel, domain.size, prepared.inputMont, prepared.stageMont, prepared.scaleMont, false, iters);
      lines.push(`${domain.size},forward_ntt,${initMs.toFixed(3)},${forward.cold.bitReverseMs.toFixed(3)},${forward.cold.stageUploadMs.toFixed(3)},${forward.cold.stageKernelMs.toFixed(3)},${forward.cold.stageReadbackMs.toFixed(3)},${forward.cold.stageTotalMs.toFixed(3)},${forward.cold.scaleMs.toFixed(3)},${forward.cold.totalMs.toFixed(3)},${(initMs + forward.cold.totalMs).toFixed(3)},${forward.warm.bitReverseMs.toFixed(3)},${forward.warm.stageTotalMs.toFixed(3)},${forward.warm.scaleMs.toFixed(3)},${forward.warm.totalMs.toFixed(3)}`);
      const inverse = await benchmarkNTT(device, vectorKernel, nttKernel, domain.size, prepared.inputMont, prepared.inverseStageMont, prepared.scaleMont, true, iters);
      lines.push(`${domain.size},inverse_ntt,${initMs.toFixed(3)},${inverse.cold.bitReverseMs.toFixed(3)},${inverse.cold.stageUploadMs.toFixed(3)},${inverse.cold.stageKernelMs.toFixed(3)},${inverse.cold.stageReadbackMs.toFixed(3)},${inverse.cold.stageTotalMs.toFixed(3)},${inverse.cold.scaleMs.toFixed(3)},${inverse.cold.totalMs.toFixed(3)},${(initMs + inverse.cold.totalMs).toFixed(3)},${inverse.warm.bitReverseMs.toFixed(3)},${inverse.warm.stageTotalMs.toFixed(3)},${inverse.warm.scaleMs.toFixed(3)},${inverse.warm.totalMs.toFixed(3)}`);
      writeLog(lines);
    }
    lines.push("");
    lines.push(`PASS: ${config.curve === "bn254" ? "BN254" : "BLS12-381"} fr NTT browser benchmark completed`);
    writeLog(lines);
    setStatus("Pass");
    setPageState("pass");
  } catch (error) {
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("Fail");
    setPageState("fail");
  } finally {
    mustElement(runButton, "run").disabled = false;
  }
}

mustElement(runButton, "run").addEventListener("click", () => {
  void runBenchmark();
});
