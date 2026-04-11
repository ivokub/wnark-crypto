export {};

type NTTConfig = {
  curve: string;
  title: string;
  supported: boolean;
  unsupportedReason?: string;
  vectorPath?: string;
  arithShaderPath?: string;
  vectorShaderPath?: string;
  nttShaderPath?: string;
  labelPrefix?: string;
};

type NTTCase = {
  name: string;
  size: number;
  input_mont_le: string[];
  forward_expected_le: string[];
  inverse_expected_le: string[];
  stage_twiddles_le: string[][];
  inverse_stage_twiddles_le: string[][];
  inverse_scale_le: string;
};

type Phase5Vectors = {
  ntt_cases: NTTCase[];
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

const FR_VECTOR_OP_MUL_FACTORS = 3;
const FR_VECTOR_OP_BIT_REVERSE_COPY = 4;
const ELEMENT_BYTES = 32;
const UNIFORM_BYTES = 32;

const CONFIGS: Record<string, NTTConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 fr NTT Browser Smoke",
    supported: true,
    vectorPath: "/testdata/vectors/fr/bn254_phase5_ntt.json",
    arithShaderPath: "/shaders/curves/bn254/fr_arith.wgsl",
    vectorShaderPath: "/shaders/curves/bn254/fr_vector.wgsl",
    nttShaderPath: "/shaders/curves/bn254/fr_ntt.wgsl",
    labelPrefix: "bn254-fr-ntt",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 fr NTT Browser Smoke",
    supported: false,
    unsupportedReason: "BLS12-381 NTT vectors and domain metadata are not implemented yet.",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;

function getConfig(): NTTConfig {
  const curve = new URLSearchParams(window.location.search).get("curve") ?? "bn254";
  const config = CONFIGS[curve];
  if (!config) {
    throw new Error(`unsupported curve: ${curve}`);
  }
  return config;
}

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function setPageState(state: "idle" | "running" | "pass" | "fail"): void {
  document.body.dataset.status = state;
}

function writeLog(lines: string[]): void {
  logEl.textContent = lines.join("\n");
}

async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors(config: NTTConfig): Promise<Phase5Vectors> {
  return JSON.parse(await fetchText(config.vectorPath!)) as Phase5Vectors;
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
}

function hexToBytes(hex: string): Uint8Array {
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
  const out = new Uint8Array(hexValues.length * ELEMENT_BYTES);
  hexValues.forEach((hex, index) => {
    out.set(hexToBytes(hex), index * ELEMENT_BYTES);
  });
  return out;
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function createKernel(device: GPUDevice, label: string, shaderCode: string, entryPoint: string): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
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
  const pipeline = device.createComputePipeline({
    label: `${label}-pipeline`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

async function runBitReverse(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  inputHex: readonly string[],
): Promise<string[]> {
  const count = inputHex.length;
  const dataBytes = count * ELEMENT_BYTES;
  const zeroHex = "0000000000000000000000000000000000000000000000000000000000000000";
  const zeros = inputHex.map(() => zeroHex);
  const inputA = createStorageBuffer(device, "input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({ label: "params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const inputBytes = packHexBatch(inputHex);
  const zeroBytes = packHexBatch(zeros);
  device.queue.writeBuffer(inputA, 0, inputBytes.buffer.slice(inputBytes.byteOffset, inputBytes.byteOffset + inputBytes.byteLength));
  device.queue.writeBuffer(inputB, 0, zeroBytes.buffer.slice(zeroBytes.byteOffset, zeroBytes.byteOffset + zeroBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = FR_VECTOR_OP_BIT_REVERSE_COPY;
  params[2] = Math.round(Math.log2(count));
  device.queue.writeBuffer(uniform, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
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

  const out: string[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(bytesToHex(view.slice(i * ELEMENT_BYTES, (i + 1) * ELEMENT_BYTES)));
  }
  return out;
}

async function runMulFactors(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  inputHex: readonly string[],
  factorHex: string,
): Promise<string[]> {
  const count = inputHex.length;
  const dataBytes = count * ELEMENT_BYTES;
  const factorsHex = inputHex.map(() => factorHex);
  const inputA = createStorageBuffer(device, "input-a", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputB = createStorageBuffer(device, "input-b", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = createStorageBuffer(device, "output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = createStorageBuffer(device, "staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniform = device.createBuffer({ label: "params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const inputBytes = packHexBatch(inputHex);
  const factorBytes = packHexBatch(factorsHex);
  device.queue.writeBuffer(inputA, 0, inputBytes.buffer.slice(inputBytes.byteOffset, inputBytes.byteOffset + inputBytes.byteLength));
  device.queue.writeBuffer(inputB, 0, factorBytes.buffer.slice(factorBytes.byteOffset, factorBytes.byteOffset + factorBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = FR_VECTOR_OP_MUL_FACTORS;
  device.queue.writeBuffer(uniform, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
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

  const out: string[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(bytesToHex(view.slice(i * ELEMENT_BYTES, (i + 1) * ELEMENT_BYTES)));
  }
  return out;
}

async function runNTTStages(
  device: GPUDevice,
  nttKernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  inputHex: readonly string[],
  stageTwiddlesHex: readonly string[][],
): Promise<string[]> {
  let state = [...inputHex];
  for (let stage = 0; stage < stageTwiddlesHex.length; stage += 1) {
    const count = state.length;
    const dataBytes = count * ELEMENT_BYTES;
    const twiddles = stageTwiddlesHex[stage];
    const twiddleBytes = packHexBatch(twiddles);
    const stateBytes = packHexBatch(state);
    const inputA = createStorageBuffer(device, "ntt-input", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const inputB = createStorageBuffer(device, "ntt-twiddles", twiddleBytes.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const output = createStorageBuffer(device, "ntt-output", dataBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const staging = createStorageBuffer(device, "ntt-staging", dataBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const uniform = device.createBuffer({ label: "ntt-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(inputA, 0, stateBytes.buffer.slice(stateBytes.byteOffset, stateBytes.byteOffset + stateBytes.byteLength));
    device.queue.writeBuffer(inputB, 0, twiddleBytes.buffer.slice(twiddleBytes.byteOffset, twiddleBytes.byteOffset + twiddleBytes.byteLength));
    const params = new Uint32Array(UNIFORM_BYTES / 4);
    params[0] = count;
    params[1] = stage;
    device.queue.writeBuffer(uniform, 0, params.buffer);

    const bindGroup = device.createBindGroup({
      layout: nttKernel.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputA } },
        { binding: 1, resource: { buffer: inputB } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: uniform } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(nttKernel.pipeline);
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

    state = [];
    for (let i = 0; i < count; i += 1) {
      state.push(bytesToHex(view.slice(i * ELEMENT_BYTES, (i + 1) * ELEMENT_BYTES)));
    }
  }
  return state;
}

function expectHexBatch(name: string, got: readonly string[], want: readonly string[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i] !== want[i]) {
      throw new Error(`${name}: mismatch at index ${i}: got=${got[i]} want=${want[i]}`);
    }
  }
}

async function runSmoke(config: NTTConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;
  try {
    if (!config.supported) {
      throw new Error(config.unsupportedReason ?? `NTT is not implemented for ${config.curve}`);
    }
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

    const [arithShader, vectorShader, nttShader, vectors] = await Promise.all([
      fetchText(config.arithShaderPath!),
      fetchText(config.vectorShaderPath!),
      fetchText(config.nttShaderPath!),
      fetchVectors(config),
    ]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`cases.ntt = ${vectors.ntt_cases.length}`);

    const arithKernel = createKernel(device, `${config.labelPrefix}-arith`, arithShader, "fr_ops_main");
    const vectorKernel = createKernel(device, `${config.labelPrefix}-vector`, vectorShader, "fr_vector_main");
    const nttKernel = createKernel(device, `${config.labelPrefix}-ntt`, nttShader, "fr_ntt_stage_main");
    lines.push("4. Creating pipeline... OK");

    for (const item of vectors.ntt_cases) {
      const bitReversed = await runBitReverse(device, vectorKernel, item.input_mont_le);
      const forward = await runNTTStages(device, nttKernel, bitReversed, item.stage_twiddles_le);
      expectHexBatch(`${item.name}: forward_ntt`, forward, item.forward_expected_le);
      const inverseBitReversed = await runBitReverse(device, vectorKernel, forward);
      const inverseStages = await runNTTStages(device, nttKernel, inverseBitReversed, item.inverse_stage_twiddles_le);
      const inverse = await runMulFactors(device, vectorKernel, inverseStages, item.inverse_scale_le);
      expectHexBatch(`${item.name}: inverse_ntt`, inverse, item.inverse_expected_le);
    }
    lines.push("forward_ntt: OK");
    lines.push("inverse_ntt: OK");
    lines.push("");
    lines.push(`PASS: ${config.curve === "bn254" ? "BN254" : "BLS12-381"} fr NTT browser smoke succeeded`);
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
  void runSmoke(getConfig());
});
