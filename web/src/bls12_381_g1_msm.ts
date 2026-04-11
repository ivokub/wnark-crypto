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

type MSMCase = {
  name: string;
  bases_affine: AffinePoint[];
  scalars_bytes_le: string[];
  expected_affine: JacobianPoint;
};

type Phase8Vectors = {
  terms_per_instance: number;
  msm_cases: MSMCase[];
  one_mont_z: string;
};

type MSMKernel = {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
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
const G1_OP_AFFINE_ADD = 7;

const FP_BYTES = 48;
const POINT_BYTES = 144;
const UNIFORM_BYTES = 32;
const INDEX_SIGN_BIT = 0x80000000;
const ZERO_HEX =
  "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

const runButton = document.getElementById("run") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const logEl = document.getElementById("log") as HTMLPreElement;

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

async function fetchVectors(): Promise<Phase8Vectors> {
  const text = await fetchText("/testdata/vectors/g1/bls12_381_phase8_msm.json?v=1");
  return JSON.parse(text) as Phase8Vectors;
}

async function getAdapterInfo(adapter: GPUAdapter): Promise<GPUAdapterInfo | null> {
  if ("info" in adapter && adapter.info) {
    return adapter.info;
  }
  const compatAdapter = adapter as GPUAdapter & {
    requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
  };
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
  if (info.vendor) {
    lines.push(`adapter.vendor = ${info.vendor}`);
  }
  if (info.architecture) {
    lines.push(`adapter.architecture = ${info.architecture}`);
  }
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

function createKernel(device: GPUDevice, shaderCode: string): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
  const shaderModule = device.createShaderModule({
    label: "bls12-381-g1-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bls12-381-g1-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bls12-381-g1-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bls12-381-g1-pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "g1_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function createMSMKernel(device: GPUDevice, shaderCode: string, entryPoint: string): MSMKernel {
  const shaderModule = device.createShaderModule({
    label: "bls12-381-g1-msm-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bls12-381-g1-msm-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bls12-381-g1-msm-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `bls12-381-g1-${entryPoint}`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

function affineToKernelPoint(point: AffinePoint, oneMontZ: string): JacobianPoint {
  const isInfinity = point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? ZERO_HEX : oneMontZ,
  };
}

function packPointBatch(points: readonly JacobianPoint[]): Uint8Array {
  const out = new Uint8Array(points.length * POINT_BYTES);
  points.forEach((point, index) => {
    const base = index * POINT_BYTES;
    out.set(hexToBytes(point.x_bytes_le), base);
    out.set(hexToBytes(point.y_bytes_le), base + FP_BYTES);
    out.set(hexToBytes(point.z_bytes_le), base + 2 * FP_BYTES);
  });
  return out;
}

function unpackPointBatch(bytes: Uint8Array, count: number): JacobianPoint[] {
  const out: JacobianPoint[] = [];
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

function zeroPoint(): JacobianPoint {
  return { x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX };
}

function jacToAffinePoint(point: JacobianPoint, oneMontZ: string): JacobianPoint {
  if (point.z_bytes_le === ZERO_HEX) {
    return zeroPoint();
  }
  return { x_bytes_le: point.x_bytes_le, y_bytes_le: point.y_bytes_le, z_bytes_le: oneMontZ };
}

async function runOp(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  opcode: number,
  inputA: readonly JacobianPoint[],
  inputB: readonly JacobianPoint[],
): Promise<JacobianPoint[]> {
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

  device.queue.writeBuffer(inputABuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  device.queue.writeBuffer(inputBBuffer, 0, bBytes.buffer.slice(bBytes.byteOffset, bBytes.byteOffset + bBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);

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

function scalarBit(scalarHexLE: string, bit: number): boolean {
  const bytes = hexToBytes(scalarHexLE);
  const byteIndex = Math.floor(bit / 8);
  const bitIndex = bit % 8;
  return ((bytes[byteIndex] >> bitIndex) & 1) !== 0;
}

function anyMask(mask: readonly boolean[]): boolean {
  return mask.some(Boolean);
}

function maskedAffine(base: readonly JacobianPoint[], mask: readonly boolean[]): JacobianPoint[] {
  return base.map((point, index) => (mask[index] ? point : zeroPoint()));
}

function scalarHexLEToBigInt(hex: string): bigint {
  const bytes = hexToBytes(hex);
  let out = 0n;
  for (let i = bytes.length - 1; i >= 0; i -= 1) {
    out = (out << 8n) | BigInt(bytes[i]);
  }
  return out;
}

function extractWindowDigit(scalar: bigint, bitOffset: number, window: number): number {
  if (window <= 0) {
    return 0;
  }
  const mask = (1n << BigInt(window)) - 1n;
  return Number((scalar >> BigInt(bitOffset)) & mask);
}

function bestPippengerWindow(count: number): number {
  const windows = [4, 5, 6, 7, 8, 9, 10, 11, 12];
  let best = windows[0];
  let bestCost = Number.POSITIVE_INFINITY;
  for (const window of windows) {
    const cost = Math.ceil(255 / window) * (count + (1 << window));
    if (cost < bestCost) {
      bestCost = cost;
      best = window;
    }
  }
  return best;
}

function buildSparseSignedBucketMetadata(
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
  window: number,
  maxChunkSize = 256,
): {
  baseIndices: Uint32Array;
  bucketPointers: Uint32Array;
  bucketSizes: Uint32Array;
  bucketValues: Uint32Array;
  windowStarts: Uint32Array;
  windowCounts: Uint32Array;
  numWindows: number;
  bucketCount: number;
} {
  const scalarBigs = scalars.map((scalar) => scalarHexLEToBigInt(scalar));
  const numWindows = Math.ceil(256 / window) + 1;
  const bucketCount = 1 << (window - 1);
  const totalWindows = count * numWindows;
  const logicalBucketSizes = new Uint32Array(totalWindows * bucketCount);
  const half = 1n << BigInt(window - 1);
  const full = 1n << BigInt(window);

  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const scalar = scalarBigs[baseOffset + term];
      let carry = 0n;
      for (let win = 0; win < numWindows; win += 1) {
        const unsigned = win < numWindows - 1 ? BigInt(extractWindowDigit(scalar, win * window, window)) : 0n;
        let value = unsigned + carry;
        carry = 0n;
        if (value >= half) {
          value = full - value;
          if (value !== 0n) {
            const slot = (instance * numWindows + win) * bucketCount + (Number(value) - 1);
            logicalBucketSizes[slot] += 1;
          }
          carry = 1n;
        } else if (value !== 0n) {
          const slot = (instance * numWindows + win) * bucketCount + (Number(value) - 1);
          logicalBucketSizes[slot] += 1;
        }
      }
    }
  }

  const logicalBucketPointers = new Uint32Array(totalWindows * bucketCount);
  let totalEntries = 0;
  for (let i = 0; i < logicalBucketSizes.length; i += 1) {
    logicalBucketPointers[i] = totalEntries;
    totalEntries += logicalBucketSizes[i];
  }
  const baseIndices = new Uint32Array(totalEntries);
  const writeOffsets = logicalBucketPointers.slice();

  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const idx = baseOffset + term;
      const scalar = scalarBigs[idx];
      let carry = 0n;
      for (let win = 0; win < numWindows; win += 1) {
        const unsigned = win < numWindows - 1 ? BigInt(extractWindowDigit(scalar, win * window, window)) : 0n;
        let value = unsigned + carry;
        carry = 0n;
        let neg = false;
        if (value >= half) {
          value = full - value;
          neg = value !== 0n;
          carry = 1n;
        }
        if (value === 0n) {
          continue;
        }
        const slot = (instance * numWindows + win) * bucketCount + (Number(value) - 1);
        const raw = neg ? ((idx | INDEX_SIGN_BIT) >>> 0) : idx;
        baseIndices[writeOffsets[slot]] = raw;
        writeOffsets[slot] += 1;
      }
    }
  }

  const bucketPointers: number[] = [];
  const bucketSizes: number[] = [];
  const bucketValues: number[] = [];
  const windowStarts = new Uint32Array(totalWindows);
  const windowCounts = new Uint32Array(totalWindows);
  for (let windowSlot = 0; windowSlot < totalWindows; windowSlot += 1) {
    windowStarts[windowSlot] = bucketPointers.length;
    let dispatchedInWindow = 0;
    const bucketBase = windowSlot * bucketCount;
    for (let value = 1; value <= bucketCount; value += 1) {
      const slot = bucketBase + (value - 1);
      const size = logicalBucketSizes[slot];
      if (size === 0) {
        continue;
      }
      const ptr = logicalBucketPointers[slot];
      for (let offset = 0; offset < size; offset += maxChunkSize) {
        bucketPointers.push(ptr + offset);
        bucketSizes.push(Math.min(size - offset, maxChunkSize));
        bucketValues.push(value);
        dispatchedInWindow += 1;
      }
    }
    windowCounts[windowSlot] = dispatchedInWindow;
  }

  return {
    baseIndices,
    bucketPointers: Uint32Array.from(bucketPointers),
    bucketSizes: Uint32Array.from(bucketSizes),
    bucketValues: Uint32Array.from(bucketValues),
    windowStarts,
    windowCounts,
    numWindows,
    bucketCount,
  };
}

async function runScalarMulAffine(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  base: readonly JacobianPoint[],
  scalarsHexLE: readonly string[],
): Promise<JacobianPoint[]> {
  let acc = await runOp(device, kernel, G1_OP_JAC_INFINITY, base.map(() => zeroPoint()), base.map(() => zeroPoint()));
  for (let bit = 255; bit >= 0; bit -= 1) {
    acc = await runOp(device, kernel, G1_OP_DOUBLE_JAC, acc, acc.map(() => zeroPoint()));
    const mask = scalarsHexLE.map((scalar) => scalarBit(scalar, bit));
    if (!anyMask(mask)) {
      continue;
    }
    acc = await runOp(device, kernel, G1_OP_ADD_MIXED, acc, maskedAffine(base, mask));
  }
  return runOp(device, kernel, G1_OP_JAC_TO_AFFINE, acc, acc.map(() => zeroPoint()));
}

async function runNaiveMSM(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  vectors: Phase8Vectors,
): Promise<JacobianPoint[]> {
  const oneMontZ = vectors.one_mont_z;
  const bases: JacobianPoint[] = [];
  const scalars: string[] = [];
  for (const item of vectors.msm_cases) {
    for (let i = 0; i < vectors.terms_per_instance; i += 1) {
      bases.push(affineToKernelPoint(item.bases_affine[i], oneMontZ));
      scalars.push(item.scalars_bytes_le[i]);
    }
  }

  const terms = await runScalarMulAffine(device, kernel, bases, scalars);
  const count = vectors.msm_cases.length;
  let width = vectors.terms_per_instance;
  let state = terms;

  while (width > 1) {
    const nextWidth = Math.floor((width + 1) / 2);
    const left: JacobianPoint[] = new Array(count * nextWidth);
    const right: JacobianPoint[] = new Array(count * nextWidth);
    for (let instance = 0; instance < count; instance += 1) {
      const rowBase = instance * width;
      const nextBase = instance * nextWidth;
      for (let j = 0; j < nextWidth; j += 1) {
        left[nextBase + j] = jacToAffinePoint(state[rowBase + 2 * j], oneMontZ);
        if (2 * j + 1 < width) {
          right[nextBase + j] = jacToAffinePoint(state[rowBase + 2 * j + 1], oneMontZ);
        } else {
          right[nextBase + j] = zeroPoint();
        }
      }
    }
    state = await runOp(device, kernel, G1_OP_AFFINE_ADD, left, right);
    width = nextWidth;
  }

  return state;
}

function createPointStorageBuffer(device: GPUDevice, label: string, points: readonly JacobianPoint[], minCount = points.length): GPUBuffer {
  const bytes = packPointBatch(points.length === 0 ? [zeroPoint()] : points);
  const count = Math.max(minCount, points.length === 0 ? 1 : points.length);
  const buffer = device.createBuffer({
    label,
    size: count * POINT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  return buffer;
}

function createU32StorageBuffer(device: GPUDevice, label: string, values: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, values.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  }
  return buffer;
}

function createEmptyPointStorageBuffer(device: GPUDevice, label: string, count: number): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(1, count) * POINT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function createParamsBuffer(
  device: GPUDevice,
  label: string,
  values: { count: number; termsPerInstance?: number; window?: number; numWindows?: number; bucketCount?: number; rowWidth?: number },
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = values.count;
  params[2] = values.termsPerInstance ?? 0;
  params[3] = values.window ?? 0;
  params[4] = values.numWindows ?? 0;
  params[5] = values.bucketCount ?? 0;
  params[6] = values.rowWidth ?? 0;
  device.queue.writeBuffer(buffer, 0, params.buffer);
  return buffer;
}

function createBindGroupForBuffers(
  device: GPUDevice,
  kernel: MSMKernel,
  label: string,
  inputA: GPUBuffer,
  inputB: GPUBuffer,
  output: GPUBuffer,
  params: GPUBuffer,
  meta0?: GPUBuffer,
  meta1?: GPUBuffer,
  meta2?: GPUBuffer,
): GPUBindGroup {
  const metaA = meta0 ?? inputB;
  const metaB = meta1 ?? inputB;
  const metaC = meta2 ?? inputB;
  return device.createBindGroup({
    label,
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: metaA } },
      { binding: 5, resource: { buffer: metaB } },
      { binding: 6, resource: { buffer: metaC } },
    ],
  });
}

async function submitKernel(device: GPUDevice, kernel: MSMKernel, bindGroup: GPUBindGroup, count: number): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "g1-msm-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-msm-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readbackPointBuffer(device: GPUDevice, buffer: GPUBuffer, count: number): Promise<JacobianPoint[]> {
  const staging = device.createBuffer({
    label: "g1-msm-readback",
    size: Math.max(1, count) * POINT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "g1-msm-readback-encoder" });
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, Math.max(1, count) * POINT_BYTES);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return unpackPointBatch(bytes, count);
}

async function runOptimizedPippengerMSM(
  device: GPUDevice,
  kernels: { bucket: MSMKernel; weightBuckets: MSMKernel; subsumPhase1: MSMKernel; combine: MSMKernel },
  bases: readonly JacobianPoint[],
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
  window: number,
): Promise<JacobianPoint[]> {
  const metadata = buildSparseSignedBucketMetadata(scalars, count, termsPerInstance, window);
  const zeroBuffer = createPointStorageBuffer(device, "g1-msm-zero", [zeroPoint()], 1);
  const basesBuffer = createPointStorageBuffer(device, "g1-msm-bases", bases);
  const baseIndicesBuffer = createU32StorageBuffer(device, "g1-msm-base-indices", metadata.baseIndices);
  const bucketPointersBuffer = createU32StorageBuffer(device, "g1-msm-bucket-pointers", metadata.bucketPointers);
  const bucketSizesBuffer = createU32StorageBuffer(device, "g1-msm-bucket-sizes", metadata.bucketSizes);
  const bucketValuesBuffer = createU32StorageBuffer(device, "g1-msm-bucket-values", metadata.bucketValues);
  const windowStartsBuffer = createU32StorageBuffer(device, "g1-msm-window-starts", metadata.windowStarts);
  const windowCountsBuffer = createU32StorageBuffer(device, "g1-msm-window-counts", metadata.windowCounts);

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBuffer(device, "g1-msm-bucket-out", bucketCountOut);
  const bucketParams = createParamsBuffer(device, "g1-msm-bucket-params", {
    count: bucketCountOut,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  await submitKernel(
    device,
    kernels.bucket,
    createBindGroupForBuffers(
      device,
      kernels.bucket,
      "g1-msm-bucket-bg",
      basesBuffer,
      zeroBuffer,
      bucketOutput,
      bucketParams,
      baseIndicesBuffer,
      bucketPointersBuffer,
      bucketSizesBuffer,
    ),
    bucketCountOut,
  );

  const weightedBucketOutput = createEmptyPointStorageBuffer(device, "g1-msm-weighted-out", bucketCountOut);
  const weightParams = createParamsBuffer(device, "g1-msm-weight-params", { count: bucketCountOut });
  await submitKernel(
    device,
    kernels.weightBuckets,
    createBindGroupForBuffers(
      device,
      kernels.weightBuckets,
      "g1-msm-weight-bg",
      bucketOutput,
      zeroBuffer,
      weightedBucketOutput,
      weightParams,
      bucketValuesBuffer,
    ),
    bucketCountOut,
  );

  const windowOutput = createEmptyPointStorageBuffer(device, "g1-msm-window-out", count * metadata.numWindows);
  const windowParams = createParamsBuffer(device, "g1-msm-window-params", { count: count * metadata.numWindows });
  await submitKernel(
    device,
    kernels.subsumPhase1,
    createBindGroupForBuffers(
      device,
      kernels.subsumPhase1,
      "g1-msm-window-bg",
      weightedBucketOutput,
      zeroBuffer,
      windowOutput,
      windowParams,
      bucketValuesBuffer,
      windowStartsBuffer,
      windowCountsBuffer,
    ),
    count * metadata.numWindows * 64,
  );

  const finalOutput = createEmptyPointStorageBuffer(device, "g1-msm-final-out", count);
  const finalParams = createParamsBuffer(device, "g1-msm-final-params", {
    count,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  await submitKernel(
    device,
    kernels.combine,
    createBindGroupForBuffers(
      device,
      kernels.combine,
      "g1-msm-combine-bg",
      windowOutput,
      zeroBuffer,
      finalOutput,
      finalParams,
    ),
    count,
  );

  const result = await readbackPointBuffer(device, finalOutput, count);

  zeroBuffer.destroy();
  basesBuffer.destroy();
  baseIndicesBuffer.destroy();
  bucketPointersBuffer.destroy();
  bucketSizesBuffer.destroy();
  bucketValuesBuffer.destroy();
  windowStartsBuffer.destroy();
  windowCountsBuffer.destroy();
  bucketOutput.destroy();
  weightedBucketOutput.destroy();
  windowOutput.destroy();
  finalOutput.destroy();
  bucketParams.destroy();
  weightParams.destroy();
  windowParams.destroy();
  finalParams.destroy();

  return result;
}

async function runSmoke(): Promise<void> {
  const lines = ["=== BLS12-381 G1 Phase 8 Browser Smoke ===", ""];
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

    const [opsShaderText, msmShaderText, vectors] = await Promise.all([
      fetchText("/shaders/curves/bls12_381/g1_arith.wgsl?v=1"),
      fetchText("/shaders/curves/bls12_381/g1_msm.wgsl?v=2"),
      fetchVectors(),
    ]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
    lines.push(`cases.msm = ${vectors.msm_cases.length}`);

    const kernel = createKernel(device, opsShaderText);
    const optimizedKernels = {
      bucket: createMSMKernel(device, msmShaderText, "g1_msm_bucket_sparse_main"),
      weightBuckets: createMSMKernel(device, msmShaderText, "g1_msm_weight_buckets_main"),
      subsumPhase1: createMSMKernel(device, msmShaderText, "g1_msm_subsum_phase1_main"),
      combine: createMSMKernel(device, msmShaderText, "g1_msm_combine_main"),
    };
    lines.push("4. Creating pipeline... OK");

    const bases: JacobianPoint[] = [];
    const scalars: string[] = [];
    const want: JacobianPoint[] = [];
    for (const item of vectors.msm_cases) {
      for (let i = 0; i < vectors.terms_per_instance; i += 1) {
        bases.push(affineToKernelPoint(item.bases_affine[i], vectors.one_mont_z));
        scalars.push(item.scalars_bytes_le[i]);
      }
      want.push(item.expected_affine);
    }

    expectPointBatch(
      "msm_naive_affine",
      await runNaiveMSM(device, kernel, vectors),
      want,
    );
    lines.push("msm_naive_affine: OK");

    const window = bestPippengerWindow(vectors.terms_per_instance);
    expectPointBatch(
      "msm_pippenger_affine",
      await runOptimizedPippengerMSM(
        device,
        optimizedKernels,
        bases,
        scalars,
        vectors.msm_cases.length,
        vectors.terms_per_instance,
        window,
      ),
      want,
    );
    lines.push(`msm_pippenger_affine (window=${window}): OK`);

    lines.push("");
    lines.push("PASS: BLS12-381 G1 Phase 8 browser smoke succeeded");
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
