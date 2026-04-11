export {};

type AffinePoint = {
  x_bytes_le: string;
  y_bytes_le: string;
};

type JacobianPoint = {
  x_bytes_le: string;
  y_bytes_le: string;
  z_bytes_le: string;
};

type ScalarMulCase = {
  name: string;
  base_affine: AffinePoint;
  scalar_bytes_le: string;
  scalar_mul_affine: JacobianPoint;
};

type BaseMulCase = {
  name: string;
  scalar_bytes_le: string;
  scalar_mul_base_affine: JacobianPoint;
};

type Phase7Vectors = {
  generator_affine: AffinePoint;
  scalar_cases: ScalarMulCase[];
  base_cases: BaseMulCase[];
};

type G1ScalarConfig = {
  curve: string;
  title: string;
  vectorPath: string;
  shaderPath: string;
  labelPrefix: string;
  fpBytes: number;
  pointBytes: number;
  zeroHex: string;
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

const G1_OP_JAC_INFINITY = 1;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const UNIFORM_BYTES = 32;

const CONFIGS: Record<string, G1ScalarConfig> = {
  bn254: {
    curve: "bn254",
    title: "BN254 G1 Phase 7 Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bn254_phase7_scalar_mul.json",
    shaderPath: "/shaders/curves/bn254/g1_arith.wgsl",
    labelPrefix: "bn254-g1",
    fpBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
  },
  bls12_381: {
    curve: "bls12_381",
    title: "BLS12-381 G1 Phase 7 Browser Smoke",
    vectorPath: "/testdata/vectors/g1/bls12_381_phase7_scalar_mul.json?v=1",
    shaderPath: "/shaders/curves/bls12_381/g1_arith.wgsl?v=1",
    labelPrefix: "bls12-381-g1",
    fpBytes: 48,
    pointBytes: 144,
    zeroHex: "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
  },
};

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;

function getConfig(): G1ScalarConfig {
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

async function fetchVectors(config: G1ScalarConfig): Promise<Phase7Vectors> {
  const text = await fetchText(config.vectorPath);
  return JSON.parse(text) as Phase7Vectors;
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

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function createKernel(device: GPUDevice, shaderCode: string, config: G1ScalarConfig): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
  const shaderModule = device.createShaderModule({
    label: `${config.labelPrefix}-shader`,
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${config.labelPrefix}-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${config.labelPrefix}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${config.labelPrefix}-pipeline`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "g1_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function affineToKernelPoint(point: AffinePoint, oneMontZ: string, zeroHex: string): JacobianPoint {
  const isInfinity = point.x_bytes_le === zeroHex && point.y_bytes_le === zeroHex;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? zeroHex : oneMontZ,
  };
}

function packPointBatch(points: readonly JacobianPoint[], config: G1ScalarConfig): Uint8Array {
  const out = new Uint8Array(points.length * config.pointBytes);
  points.forEach((point, index) => {
    const base = index * config.pointBytes;
    out.set(hexToBytes(point.x_bytes_le), base);
    out.set(hexToBytes(point.y_bytes_le), base + config.fpBytes);
    out.set(hexToBytes(point.z_bytes_le), base + 2 * config.fpBytes);
  });
  return out;
}

function unpackPointBatch(bytes: Uint8Array, count: number, config: G1ScalarConfig): JacobianPoint[] {
  const out: JacobianPoint[] = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * config.pointBytes;
    out.push({
      x_bytes_le: bytesToHex(bytes.slice(base, base + config.fpBytes)),
      y_bytes_le: bytesToHex(bytes.slice(base + config.fpBytes, base + 2 * config.fpBytes)),
      z_bytes_le: bytesToHex(bytes.slice(base + 2 * config.fpBytes, base + 3 * config.fpBytes)),
    });
  }
  return out;
}

function findOneMontZ(vectors: Phase7Vectors, zeroHex: string): string {
  for (const scalarCase of vectors.scalar_cases) {
    if (scalarCase.scalar_mul_affine.z_bytes_le !== zeroHex) {
      return scalarCase.scalar_mul_affine.z_bytes_le;
    }
  }
  for (const baseCase of vectors.base_cases) {
    if (baseCase.scalar_mul_base_affine.z_bytes_le !== zeroHex) {
      return baseCase.scalar_mul_base_affine.z_bytes_le;
    }
  }
  throw new Error("missing non-zero affine flag in vectors");
}

function zeroPoint(config: G1ScalarConfig): JacobianPoint {
  return { x_bytes_le: config.zeroHex, y_bytes_le: config.zeroHex, z_bytes_le: config.zeroHex };
}

function scalarBit(scalarHexLE: string, bit: number): boolean {
  const bytes = hexToBytes(scalarHexLE);
  const byteIndex = Math.floor(bit / 8);
  const bitIndex = bit % 8;
  return ((bytes[byteIndex] >> bitIndex) & 1) !== 0;
}

function anyMask(mask: readonly boolean[]): boolean {
  return mask.some(Boolean);
}

function maskedAffine(base: readonly JacobianPoint[], mask: readonly boolean[], config: G1ScalarConfig): JacobianPoint[] {
  return base.map((point, index) => (mask[index] ? point : zeroPoint(config)));
}

function expectPointBatch(name: string, got: readonly JacobianPoint[], want: readonly JacobianPoint[]): void {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (
      got[i].x_bytes_le !== want[i].x_bytes_le ||
      got[i].y_bytes_le !== want[i].y_bytes_le ||
      got[i].z_bytes_le !== want[i].z_bytes_le
    ) {
      throw new Error(`${name}: mismatch at index ${i}`);
    }
  }
}

async function runOp(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  config: G1ScalarConfig,
  opcode: number,
  inputA: readonly JacobianPoint[],
  inputB: readonly JacobianPoint[],
): Promise<JacobianPoint[]> {
  const count = inputA.length;
  const aBytes = packPointBatch(inputA, config);
  const bBytes = packPointBatch(inputB, config);
  const byteSize = count * config.pointBytes;

  const inputABuffer = createStorageBuffer(device, `${config.labelPrefix}-input-a`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const inputBBuffer = createStorageBuffer(device, `${config.labelPrefix}-input-b`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createStorageBuffer(device, `${config.labelPrefix}-output`, byteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const stagingBuffer = createStorageBuffer(device, `${config.labelPrefix}-staging`, byteSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const uniformBuffer = device.createBuffer({
    label: `${config.labelPrefix}-params`,
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputABuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  device.queue.writeBuffer(inputBBuffer, 0, bBytes.buffer.slice(bBytes.byteOffset, bBytes.byteOffset + bBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    label: `${config.labelPrefix}-bind-group`,
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputABuffer } },
      { binding: 1, resource: { buffer: inputBBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: `${config.labelPrefix}-encoder` });
  const pass = encoder.beginComputePass({ label: `${config.labelPrefix}-pass` });
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

  return unpackPointBatch(result, count, config);
}

async function runScalarMulAffine(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  config: G1ScalarConfig,
  base: readonly JacobianPoint[],
  scalarsHexLE: readonly string[],
): Promise<JacobianPoint[]> {
  let acc = await runOp(device, kernel, config, G1_OP_JAC_INFINITY, base.map(() => zeroPoint(config)), base.map(() => zeroPoint(config)));
  for (let bit = 255; bit >= 0; bit -= 1) {
    acc = await runOp(device, kernel, config, G1_OP_DOUBLE_JAC, acc, acc.map(() => zeroPoint(config)));
    const mask = scalarsHexLE.map((scalar) => scalarBit(scalar, bit));
    if (!anyMask(mask)) {
      continue;
    }
    acc = await runOp(device, kernel, config, G1_OP_ADD_MIXED, acc, maskedAffine(base, mask, config));
  }
  return runOp(device, kernel, config, G1_OP_JAC_TO_AFFINE, acc, acc.map(() => zeroPoint(config)));
}

async function runSmoke(config: G1ScalarConfig): Promise<void> {
  const lines = [`=== ${config.title} ===`, ""];
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

    const [shaderText, vectors] = await Promise.all([fetchText(config.shaderPath), fetchVectors(config)]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`cases.scalar = ${vectors.scalar_cases.length}`);
    lines.push(`cases.base = ${vectors.base_cases.length}`);

    const kernel = createKernel(device, shaderText, config);
    lines.push("4. Creating pipeline... OK");

    const oneMontZ = findOneMontZ(vectors, config.zeroHex);
    const scalarBaseInputs = vectors.scalar_cases.map((item) => affineToKernelPoint(item.base_affine, oneMontZ, config.zeroHex));
    const scalarWant = vectors.scalar_cases.map((item) => item.scalar_mul_affine);

    expectPointBatch(
      "scalar_mul_affine",
      await runScalarMulAffine(device, kernel, config, scalarBaseInputs, vectors.scalar_cases.map((item) => item.scalar_bytes_le)),
      scalarWant,
    );
    lines.push("scalar_mul_affine: OK");

    const generatorKernel = affineToKernelPoint(vectors.generator_affine, oneMontZ, config.zeroHex);
    const baseInputs = vectors.base_cases.map(() => generatorKernel);
    const baseWant = vectors.base_cases.map((item) => item.scalar_mul_base_affine);

    expectPointBatch(
      "scalar_mul_base_affine",
      await runScalarMulAffine(device, kernel, config, baseInputs, vectors.base_cases.map((item) => item.scalar_bytes_le)),
      baseWant,
    );
    lines.push("scalar_mul_base_affine: OK");

    lines.push("");
    lines.push(`PASS: ${config.curve === "bn254" ? "BN254" : "BLS12-381"} G1 Phase 7 browser smoke succeeded`);
    writeLog(lines);
    setStatus("PASS");
    setPageState("pass");
  } catch (error) {
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("FAIL");
    setPageState("fail");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void runSmoke(getConfig());
});
