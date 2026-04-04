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

type Phase7Vectors = {
  generator_affine: AffinePoint;
  one_mont_z: string;
};

type Kernel = {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
};

type OpProfile = {
  uploadMs: number;
  kernelMs: number;
  readbackMs: number;
  totalMs: number;
};

type GpuPointBatch = {
  count: number;
  storage: GPUBuffer;
};

type ScalarBatch = {
  hexes: string[];
  words: Uint32Array;
};

type MSMProfile = {
  partitionMs: number;
  uploadMs: number;
  kernelMs: number;
  readbackMs: number;
  scalarMulTotalMs: number;
  bucketReductionTotalMs: number;
  windowReductionTotalMs: number;
  finalReductionTotalMs: number;
  reductionTotalMs: number;
  totalMs: number;
};

declare const GPUShaderStage: { COMPUTE: number };
declare const GPUBufferUsage: { STORAGE: number; COPY_DST: number; COPY_SRC: number; MAP_READ: number; UNIFORM: number };
declare const GPUMapMode: { READ: number };

const FP_BYTES = 32;
const POINT_BYTES = 96;
const UNIFORM_BYTES = 32;
const ZERO_HEX = "0000000000000000000000000000000000000000000000000000000000000000";
const INDEX_SIGN_BIT = 0x80000000;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;
const G1_OP_SCALAR_MUL_AFFINE = 8;

const minLogEl = document.getElementById("min-log") as HTMLInputElement | null;
const maxLogEl = document.getElementById("max-log") as HTMLInputElement | null;
const itersEl = document.getElementById("iters") as HTMLInputElement | null;
const runButton = document.getElementById("run") as HTMLButtonElement | null;
const statusEl = document.getElementById("status") as HTMLElement | null;
const logEl = document.getElementById("log") as HTMLElement | null;

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
  const adapterWithInfo = adapter as GPUAdapter & {
    info?: GPUAdapterInfo;
    requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
  };
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
  if (info.vendor) {
    lines.push(`adapter.vendor = ${info.vendor}`);
  }
  if (info.architecture) {
    lines.push(`adapter.architecture = ${info.architecture}`);
  }
}

function createKernel(device: GPUDevice, shaderCode: string, entryPoint = "g1_ops_main"): Kernel {
  const shaderModule = device.createShaderModule({ label: "bn254-g1-shader", code: shaderCode });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-g1-bgl",
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
    label: "bn254-g1-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-g1-pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
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

function zeroPoint(): JacobianPoint {
  return { x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX };
}

function isInfinityPoint(point: JacobianPoint): boolean {
  return point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
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

function extractWindowDigitWords(words: Uint32Array, scalarBase: number, bitOffset: number, window: number): number {
  if (window <= 0) {
    return 0;
  }
  const word = Math.floor(bitOffset / 32);
  const shift = bitOffset % 32;
  const mask = (1 << window) - 1;
  if (word >= 8) {
    return 0;
  }
  const lo = words[scalarBase + word] >>> shift;
  if (shift + window <= 32 || word + 1 >= 8) {
    return lo & mask;
  }
  const highWidth = shift + window - 32;
  const hiMask = (1 << highWidth) - 1;
  const hi = words[scalarBase + word + 1] & hiMask;
  return (lo | (hi << (32 - shift))) & mask;
}

function buildSparseBucketMetadata(
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
  window: number,
): {
  baseIndices: Uint32Array;
  bucketPointers: Uint32Array;
  bucketSizes: Uint32Array;
} {
  const numWindows = Math.ceil(256 / window);
  const bucketCount = (1 << window) - 1;
  const totalBuckets = count * numWindows * bucketCount;
  const bucketSizes = new Uint32Array(totalBuckets);
  const scalarBigs = scalars.map((scalar) => scalarHexLEToBigInt(scalar));
  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const idx = baseOffset + term;
      const scalar = scalarBigs[idx];
      for (let win = 0; win < numWindows; win += 1) {
        const digit = extractWindowDigit(scalar, win * window, window);
        if (digit === 0) {
          continue;
        }
        const bucketIndex = ((instance * numWindows + win) * bucketCount) + (digit - 1);
        bucketSizes[bucketIndex] += 1;
      }
    }
  }
  const bucketPointers = new Uint32Array(totalBuckets);
  let totalEntries = 0;
  for (let i = 0; i < totalBuckets; i += 1) {
    bucketPointers[i] = totalEntries;
    totalEntries += bucketSizes[i];
  }
  const fill = new Uint32Array(totalBuckets);
  const baseIndices = new Uint32Array(totalEntries);
  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const idx = baseOffset + term;
      const scalar = scalarBigs[idx];
      for (let win = 0; win < numWindows; win += 1) {
        const digit = extractWindowDigit(scalar, win * window, window);
        if (digit === 0) {
          continue;
        }
        const bucketIndex = ((instance * numWindows + win) * bucketCount) + (digit - 1);
        const entryOffset = bucketPointers[bucketIndex] + fill[bucketIndex];
        baseIndices[entryOffset] = idx;
        fill[bucketIndex] += 1;
      }
    }
  }
  return { baseIndices, bucketPointers, bucketSizes };
}

function buildSparseSignedBucketMetadata(
  scalarWords: Uint32Array,
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
  const numWindows = Math.ceil(256 / window) + 1;
  const bucketCount = 1 << (window - 1);
  const totalWindows = count * numWindows;
  const logicalBucketSizes = new Uint32Array(totalWindows * bucketCount);
  const half = 1 << (window - 1);
  const full = 1 << window;
  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const idx = baseOffset + term;
      const scalarBase = idx * 8;
      let carry = 0;
      for (let win = 0; win < numWindows; win += 1) {
        const unsigned = win < numWindows - 1 ? extractWindowDigitWords(scalarWords, scalarBase, win * window, window) : 0;
        let value = unsigned + carry;
        carry = 0;
        if (value >= half) {
          value = full - value;
          if (value !== 0) {
            const slot = (instance * numWindows + win) * bucketCount + (value - 1);
            logicalBucketSizes[slot] += 1;
          }
          carry = 1;
        } else if (value !== 0) {
          const slot = (instance * numWindows + win) * bucketCount + (value - 1);
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
      const scalarBase = idx * 8;
      let carry = 0;
      for (let win = 0; win < numWindows; win += 1) {
        const unsigned = win < numWindows - 1 ? extractWindowDigitWords(scalarWords, scalarBase, win * window, window) : 0;
        let value = unsigned + carry;
        carry = 0;
        let neg = false;
        if (value >= half) {
          value = full - value;
          neg = value !== 0;
          carry = 1;
        }
        if (value === 0) {
          continue;
        }
        const slot = (instance * numWindows + win) * bucketCount + (value - 1);
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

function affineToKernelPoint(point: AffinePoint, oneMontZ: string): JacobianPoint {
  const isInfinity = point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? ZERO_HEX : oneMontZ,
  };
}

function scalarToKernelPoint(scalarHexLE: string): JacobianPoint {
  return {
    x_bytes_le: scalarHexLE,
    y_bytes_le: ZERO_HEX,
    z_bytes_le: ZERO_HEX,
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

function makeScalarHexLEFromUint64(value: bigint): string {
  const out = new Uint8Array(32);
  let x = value;
  for (let i = 0; i < 8; i += 1) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return bytesToHex(out);
}

function makeRandomScalarHexLE(seed: number): string {
  const out = new Uint8Array(32);
  let state = seed >>> 0;
  for (let i = 0; i < out.length; i += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    out[i] = state & 0xff;
  }
  return bytesToHex(out);
}

function makeRandomScalarData(seed: number): { hex: string; words: Uint32Array } {
  const bytes = new Uint8Array(32);
  const words = new Uint32Array(8);
  let state = seed >>> 0;
  for (let i = 0; i < bytes.length; i += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    const value = state & 0xff;
    bytes[i] = value;
    words[i >>> 2] |= value << ((i & 3) * 8);
  }
  return { hex: bytesToHex(bytes), words };
}

async function runOpProfiled(
  device: GPUDevice,
  kernel: Kernel,
  opcode: number,
  inputA: readonly JacobianPoint[],
  inputB: readonly JacobianPoint[],
  extraParams?: {
    count?: number;
    termsPerInstance?: number;
    window?: number;
    numWindows?: number;
    bucketCount?: number;
    rowWidth?: number;
  },
): Promise<{ out: JacobianPoint[]; profile: OpProfile }> {
  const count = extraParams?.count ?? inputA.length;
  if (count === 0) {
    return {
      out: [],
      profile: { uploadMs: 0, kernelMs: 0, readbackMs: 0, totalMs: 0 },
    };
  }
  const aBytes = packPointBatch(inputA);
  const bBytes = packPointBatch(inputB.length === 0 ? [zeroPoint()] : inputB);
  const inputCount = Math.max(inputA.length, inputB.length === 0 ? 1 : inputB.length, count);
  const inputByteSize = inputCount * POINT_BYTES;
  const outputByteSize = count * POINT_BYTES;
  const totalStart = performance.now();

  const basesBuffer = device.createBuffer({ label: "g1-input-a", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const scalarsBuffer = device.createBuffer({ label: "g1-input-b", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ label: "g1-output", size: outputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ label: "g1-staging", size: outputByteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniformBuffer = device.createBuffer({ label: "g1-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const metaDummyBuffer = device.createBuffer({ label: "g1-meta-dummy", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const uploadStart = performance.now();
  device.queue.writeBuffer(basesBuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  device.queue.writeBuffer(scalarsBuffer, 0, bBytes.buffer.slice(bBytes.byteOffset, bBytes.byteOffset + bBytes.byteLength));
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  params[2] = extraParams?.termsPerInstance ?? 0;
  params[3] = extraParams?.window ?? 0;
  params[4] = extraParams?.numWindows ?? 0;
  params[5] = extraParams?.bucketCount ?? 0;
  params[6] = extraParams?.rowWidth ?? 0;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);
  const bindGroup = device.createBindGroup({
    label: "g1-bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: basesBuffer } },
      { binding: 1, resource: { buffer: scalarsBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: metaDummyBuffer } },
      { binding: 5, resource: { buffer: metaDummyBuffer } },
      { binding: 6, resource: { buffer: metaDummyBuffer } },
    ],
  });
  const uploadMs = performance.now() - uploadStart;

  const kernelStart = performance.now();
  const encoder = device.createCommandEncoder({ label: "g1-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteSize);
  device.queue.submit([encoder.finish()]);
  const kernelMs = performance.now() - kernelStart;

  const readbackStart = performance.now();
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();
  const readbackMs = performance.now() - readbackStart;

  basesBuffer.destroy();
  scalarsBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  metaDummyBuffer.destroy();

  return {
    out: unpackPointBatch(out, count),
    profile: {
      uploadMs,
      kernelMs,
      readbackMs,
      totalMs: performance.now() - totalStart,
    },
  };
}

function createPointStorageBuffer(
  device: GPUDevice,
  label: string,
  points: readonly JacobianPoint[],
  minCount = points.length,
): { buffer: GPUBuffer; uploadMs: number } {
  const bytes = packPointBatch(points.length === 0 ? [zeroPoint()] : points);
  const count = Math.max(minCount, points.length === 0 ? 1 : points.length);
  const size = count * POINT_BYTES;
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const start = performance.now();
  device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  return { buffer, uploadMs: performance.now() - start };
}

function createU32StorageBuffer(
  device: GPUDevice,
  label: string,
  values: Uint32Array,
): { buffer: GPUBuffer; uploadMs: number } {
  const byteLength = Math.max(4, values.byteLength);
  const buffer = device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const uploadStart = performance.now();
  if (values.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  }
  return { buffer, uploadMs: performance.now() - uploadStart };
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
  values: {
    count: number;
    opcode?: number;
    termsPerInstance?: number;
    window?: number;
    numWindows?: number;
    bucketCount?: number;
    rowWidth?: number;
  },
): { buffer: GPUBuffer; uploadMs: number } {
  const buffer = device.createBuffer({
    label,
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = values.count;
  params[1] = values.opcode ?? 0;
  params[2] = values.termsPerInstance ?? 0;
  params[3] = values.window ?? 0;
  params[4] = values.numWindows ?? 0;
  params[5] = values.bucketCount ?? 0;
  params[6] = values.rowWidth ?? 0;
  const start = performance.now();
  device.queue.writeBuffer(buffer, 0, params.buffer);
  return { buffer, uploadMs: performance.now() - start };
}

function createBindGroupForBuffers(
  device: GPUDevice,
  kernel: Kernel,
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

async function submitKernelProfiled(
  device: GPUDevice,
  kernel: Kernel,
  bindGroup: GPUBindGroup,
  count: number,
  label: string,
): Promise<number> {
  const start = performance.now();
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  const pass = encoder.beginComputePass({ label: `${label}-pass` });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - start;
}

async function readbackPointBufferProfiled(
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<{ points: JacobianPoint[]; readbackMs: number }> {
  const size = Math.max(1, count) * POINT_BYTES;
  const staging = device.createBuffer({
    label: "g1-readback-staging",
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const start = performance.now();
  const encoder = device.createCommandEncoder({ label: "g1-readback-encoder" });
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return { points: unpackPointBatch(bytes, count), readbackMs: performance.now() - start };
}

async function buildBases(
  device: GPUDevice,
  kernel: Kernel,
  generator: AffinePoint,
  oneMontZ: string,
  count: number,
): Promise<JacobianPoint[]> {
  const generatorKernel = affineToKernelPoint(generator, oneMontZ);
  const bases = Array.from({ length: count }, () => generatorKernel);
  const scalars = Array.from({ length: count }, (_, index) => scalarToKernelPoint(makeScalarHexLEFromUint64(BigInt(index + 1))));
  return (await runOpProfiled(device, kernel, G1_OP_SCALAR_MUL_AFFINE, bases, scalars)).out;
}

function makeMSMScalars(count: number): ScalarBatch {
  const hexes = new Array<string>(count);
  const words = new Uint32Array(count * 8);
  for (let index = 0; index < count; index += 1) {
    const scalar = makeRandomScalarData((0x9e3779b9 ^ count ^ index) >>> 0);
    hexes[index] = scalar.hex;
    words.set(scalar.words, index * 8);
  }
  return { hexes, words };
}

async function runMSMProfiled(
  device: GPUDevice,
  kernel: Kernel,
  bases: readonly JacobianPoint[],
  scalars: readonly string[],
): Promise<MSMProfile> {
  const totalStart = performance.now();
  let uploadMs = 0;
  let kernelMs = 0;
  let readbackMs = 0;
  let scalarMulTotalMs = 0;
  let reductionTotalMs = 0;

  const scalarInputs = scalars.map((scalar) => scalarToKernelPoint(scalar));
  let stage = await runOpProfiled(device, kernel, G1_OP_SCALAR_MUL_AFFINE, bases, scalarInputs);
  uploadMs += stage.profile.uploadMs;
  kernelMs += stage.profile.kernelMs;
  readbackMs += stage.profile.readbackMs;
  scalarMulTotalMs += stage.profile.totalMs;

  let width = bases.length;
  let state = stage.out;
  while (width > 1) {
    const nextWidth = Math.ceil(width / 2);
    const left: JacobianPoint[] = [];
    const right: JacobianPoint[] = [];
    for (let j = 0; j < nextWidth; j += 1) {
      left.push(state[2 * j]);
      right.push(2 * j + 1 < width ? state[2 * j + 1] : { x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX });
    }
    stage = await runOpProfiled(device, kernel, G1_OP_AFFINE_ADD, left, right);
    uploadMs += stage.profile.uploadMs;
    kernelMs += stage.profile.kernelMs;
    readbackMs += stage.profile.readbackMs;
    reductionTotalMs += stage.profile.totalMs;
    state = stage.out;
    width = nextWidth;
  }

  return {
    partitionMs: 0,
    uploadMs,
    kernelMs,
    readbackMs,
    scalarMulTotalMs,
    bucketReductionTotalMs: reductionTotalMs,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs,
    totalMs: performance.now() - totalStart,
  };
}

async function reduceAffineBucketsProfiled(
  device: GPUDevice,
  kernel: Kernel,
  bucketLists: JacobianPoint[][],
): Promise<{ out: JacobianPoint[]; profile: MSMProfile }> {
  const profile: MSMProfile = {
    partitionMs: 0,
    uploadMs: 0,
    kernelMs: 0,
    readbackMs: 0,
    scalarMulTotalMs: 0,
    bucketReductionTotalMs: 0,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs: 0,
    totalMs: 0,
  };
  const totalStart = performance.now();
  let current = bucketLists.map((bucket) => bucket.slice());
  for (;;) {
    const next: JacobianPoint[][] = Array.from({ length: current.length }, () => []);
    const left: JacobianPoint[] = [];
    const right: JacobianPoint[] = [];
    const mappings: Array<{ bucket: number; slot: number }> = [];
    let work = false;
    for (let bucketIndex = 0; bucketIndex < current.length; bucketIndex += 1) {
      const bucket = current[bucketIndex];
      if (bucket.length <= 1) {
        next[bucketIndex] = bucket.slice();
        continue;
      }
      work = true;
      const nextCount = Math.ceil(bucket.length / 2);
      next[bucketIndex] = new Array<JacobianPoint>(nextCount);
      for (let j = 0; j < nextCount; j += 1) {
        left.push(bucket[2 * j]);
        right.push(2 * j + 1 < bucket.length ? bucket[2 * j + 1] : zeroPoint());
        mappings.push({ bucket: bucketIndex, slot: j });
      }
    }
    if (!work) {
      profile.reductionTotalMs += profile.bucketReductionTotalMs;
      profile.totalMs = performance.now() - totalStart;
      return {
        out: current.map((bucket) => (bucket.length === 1 ? bucket[0] : zeroPoint())),
        profile,
      };
    }
    const reduced = await runOpProfiled(device, kernel, G1_OP_AFFINE_ADD, left, right);
    profile.uploadMs += reduced.profile.uploadMs;
    profile.kernelMs += reduced.profile.kernelMs;
    profile.readbackMs += reduced.profile.readbackMs;
    profile.bucketReductionTotalMs += reduced.profile.totalMs;
    for (let i = 0; i < reduced.out.length; i += 1) {
      const mapping = mappings[i];
      next[mapping.bucket][mapping.slot] = reduced.out[i];
    }
    current = next;
  }
}

async function reduceWindowsProfiled(
  device: GPUDevice,
  kernel: Kernel,
  bucketSums: readonly JacobianPoint[],
  count: number,
  numWindows: number,
  bucketCount: number,
): Promise<{ out: JacobianPoint[]; profile: MSMProfile }> {
  const profile: MSMProfile = {
    partitionMs: 0,
    uploadMs: 0,
    kernelMs: 0,
    readbackMs: 0,
    scalarMulTotalMs: 0,
    bucketReductionTotalMs: 0,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs: 0,
    totalMs: 0,
  };
  const totalStart = performance.now();
  const totalSlots = count * numWindows;
  const running = Array.from({ length: totalSlots }, () => zeroPoint());
  const totals = Array.from({ length: totalSlots }, () => zeroPoint());

  for (let bucket = bucketCount - 1; bucket >= 0; bucket -= 1) {
    const activeRunningIdx: number[] = [];
    const activeRunning: JacobianPoint[] = [];
    const activeBuckets: JacobianPoint[] = [];
    for (let slot = 0; slot < totalSlots; slot += 1) {
      const point = bucketSums[slot * bucketCount + bucket];
      if (isInfinityPoint(point)) {
        continue;
      }
      activeRunningIdx.push(slot);
      activeRunning.push(running[slot]);
      activeBuckets.push(point);
    }
    if (activeRunningIdx.length > 0) {
      const nextRunning = await runOpProfiled(device, kernel, G1_OP_AFFINE_ADD, activeRunning, activeBuckets);
      profile.uploadMs += nextRunning.profile.uploadMs;
      profile.kernelMs += nextRunning.profile.kernelMs;
      profile.readbackMs += nextRunning.profile.readbackMs;
      profile.windowReductionTotalMs += nextRunning.profile.totalMs;
      for (let i = 0; i < activeRunningIdx.length; i += 1) {
        running[activeRunningIdx[i]] = nextRunning.out[i];
      }
    }

    const activeTotalIdx: number[] = [];
    const activeTotals: JacobianPoint[] = [];
    const activeRunningTotals: JacobianPoint[] = [];
    for (let slot = 0; slot < totalSlots; slot += 1) {
      if (isInfinityPoint(running[slot])) {
        continue;
      }
      activeTotalIdx.push(slot);
      activeTotals.push(totals[slot]);
      activeRunningTotals.push(running[slot]);
    }
    if (activeTotalIdx.length > 0) {
      const nextTotals = await runOpProfiled(device, kernel, G1_OP_AFFINE_ADD, activeTotals, activeRunningTotals);
      profile.uploadMs += nextTotals.profile.uploadMs;
      profile.kernelMs += nextTotals.profile.kernelMs;
      profile.readbackMs += nextTotals.profile.readbackMs;
      profile.windowReductionTotalMs += nextTotals.profile.totalMs;
      for (let i = 0; i < activeTotalIdx.length; i += 1) {
        totals[activeTotalIdx[i]] = nextTotals.out[i];
      }
    }
  }

  profile.reductionTotalMs = profile.windowReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return { out: totals, profile };
}

async function combineWindowsProfiled(
  device: GPUDevice,
  kernel: Kernel,
  windowSums: readonly JacobianPoint[],
  count: number,
  numWindows: number,
  window: number,
): Promise<{ out: JacobianPoint[]; profile: MSMProfile }> {
  const profile: MSMProfile = {
    partitionMs: 0,
    uploadMs: 0,
    kernelMs: 0,
    readbackMs: 0,
    scalarMulTotalMs: 0,
    bucketReductionTotalMs: 0,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs: 0,
    totalMs: 0,
  };
  const totalStart = performance.now();
  let acc = Array.from({ length: count }, () => zeroPoint());
  const zeros = Array.from({ length: count }, () => zeroPoint());

  for (let win = numWindows - 1; win >= 0; win -= 1) {
    if (win !== numWindows - 1) {
      for (let step = 0; step < window; step += 1) {
        const doubled = await runOpProfiled(device, kernel, G1_OP_DOUBLE_JAC, acc, zeros);
        profile.uploadMs += doubled.profile.uploadMs;
        profile.kernelMs += doubled.profile.kernelMs;
        profile.readbackMs += doubled.profile.readbackMs;
        profile.finalReductionTotalMs += doubled.profile.totalMs;
        acc = doubled.out;
      }
    }

    const activeIdx: number[] = [];
    const activeAcc: JacobianPoint[] = [];
    const activeAff: JacobianPoint[] = [];
    for (let instance = 0; instance < count; instance += 1) {
      const point = windowSums[instance * numWindows + win];
      if (isInfinityPoint(point)) {
        continue;
      }
      activeIdx.push(instance);
      activeAcc.push(acc[instance]);
      activeAff.push(point);
    }
    if (activeIdx.length > 0) {
      const nextAcc = await runOpProfiled(device, kernel, G1_OP_ADD_MIXED, activeAcc, activeAff);
      profile.uploadMs += nextAcc.profile.uploadMs;
      profile.kernelMs += nextAcc.profile.kernelMs;
      profile.readbackMs += nextAcc.profile.readbackMs;
      profile.finalReductionTotalMs += nextAcc.profile.totalMs;
      for (let i = 0; i < activeIdx.length; i += 1) {
        acc[activeIdx[i]] = nextAcc.out[i];
      }
    }
  }

  const affine = await runOpProfiled(device, kernel, G1_OP_JAC_TO_AFFINE, acc, zeros);
  profile.uploadMs += affine.profile.uploadMs;
  profile.kernelMs += affine.profile.kernelMs;
  profile.readbackMs += affine.profile.readbackMs;
  profile.finalReductionTotalMs += affine.profile.totalMs;
  profile.reductionTotalMs = profile.finalReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return { out: affine.out, profile };
}

async function runPippengerMSMProfiled(
  device: GPUDevice,
  kernels: {
    bucket: Kernel;
    windowSparse: Kernel;
    combine: Kernel;
  },
  bases: readonly JacobianPoint[],
  scalars: ScalarBatch,
  window: number,
): Promise<MSMProfile> {
  const totalStart = performance.now();
  const count = 1;
  const termsPerInstance = bases.length;
  const profile: MSMProfile = {
    partitionMs: 0,
    uploadMs: 0,
    kernelMs: 0,
    readbackMs: 0,
    scalarMulTotalMs: 0,
    bucketReductionTotalMs: 0,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs: 0,
    totalMs: 0,
  };

  const partitionStart = performance.now();
  const metadata = buildSparseSignedBucketMetadata(scalars.words, count, termsPerInstance, window);
  profile.partitionMs = performance.now() - partitionStart;

  const zeroBatch = [zeroPoint()];
  const basesInput = createPointStorageBuffer(device, "g1-pip-bases", bases);
  const zeroInput = createPointStorageBuffer(device, "g1-pip-zero", zeroBatch, 1);
  const baseIndicesInput = createU32StorageBuffer(device, "g1-pip-base-indices", metadata.baseIndices);
  const bucketPointersInput = createU32StorageBuffer(device, "g1-pip-bucket-pointers", metadata.bucketPointers);
  const bucketSizesInput = createU32StorageBuffer(device, "g1-pip-bucket-sizes", metadata.bucketSizes);
  profile.uploadMs += basesInput.uploadMs + zeroInput.uploadMs + baseIndicesInput.uploadMs + bucketPointersInput.uploadMs + bucketSizesInput.uploadMs;

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBuffer(device, "g1-pip-bucket-out", bucketCountOut);
  const bucketParams = createParamsBuffer(device, "g1-pip-bucket-params", {
    count: bucketCountOut,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  profile.uploadMs += bucketParams.uploadMs;
  const bucketBindGroup = createBindGroupForBuffers(
    device,
    kernels.bucket,
    "g1-pip-bucket-bg",
    basesInput.buffer,
    zeroInput.buffer,
    bucketOutput,
    bucketParams.buffer,
    baseIndicesInput.buffer,
    bucketPointersInput.buffer,
    bucketSizesInput.buffer,
  );
  const bucketStart = performance.now();
  const bucketKernelMs = await submitKernelProfiled(device, kernels.bucket, bucketBindGroup, bucketCountOut, "g1-pip-bucket");
  profile.kernelMs += bucketKernelMs;
  profile.bucketReductionTotalMs += performance.now() - bucketStart;

  const windowOutput = createEmptyPointStorageBuffer(device, "g1-pip-window-sparse-out", count * metadata.numWindows);
  const bucketValuesInput = createU32StorageBuffer(device, "g1-pip-window-values", metadata.bucketValues);
  const windowStartsInput = createU32StorageBuffer(device, "g1-pip-window-starts", metadata.windowStarts);
  const windowCountsInput = createU32StorageBuffer(device, "g1-pip-window-counts", metadata.windowCounts);
  const windowParams = createParamsBuffer(device, "g1-pip-window-sparse-params", {
    count: count * metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  profile.uploadMs += bucketValuesInput.uploadMs + windowStartsInput.uploadMs + windowCountsInput.uploadMs + windowParams.uploadMs;
  const windowBindGroup = createBindGroupForBuffers(
    device,
    kernels.windowSparse,
    "g1-pip-window-sparse-bg",
    bucketOutput,
    zeroInput.buffer,
    windowOutput,
    windowParams.buffer,
    bucketValuesInput.buffer,
    windowStartsInput.buffer,
    windowCountsInput.buffer,
  );
  const windowStart = performance.now();
  const windowKernelMs = await submitKernelProfiled(device, kernels.windowSparse, windowBindGroup, count * metadata.numWindows * 64, "g1-pip-window-sparse");
  profile.kernelMs += windowKernelMs;
  profile.windowReductionTotalMs += performance.now() - windowStart;

  const finalOutput = createEmptyPointStorageBuffer(device, "g1-pip-final-out", count);
  const finalParams = createParamsBuffer(device, "g1-pip-final-params", {
    count,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  profile.uploadMs += finalParams.uploadMs;
  const finalBindGroup = createBindGroupForBuffers(
    device,
    kernels.combine,
    "g1-pip-final-bg",
    windowOutput,
    zeroInput.buffer,
    finalOutput,
    finalParams.buffer,
  );
  const finalStart = performance.now();
  const finalKernelMs = await submitKernelProfiled(device, kernels.combine, finalBindGroup, count, "g1-pip-final");
  profile.kernelMs += finalKernelMs;
  profile.finalReductionTotalMs += performance.now() - finalStart;

  const readback = await readbackPointBufferProfiled(device, finalOutput, count);
  profile.readbackMs += readback.readbackMs;

  basesInput.buffer.destroy();
  zeroInput.buffer.destroy();
  baseIndicesInput.buffer.destroy();
  bucketPointersInput.buffer.destroy();
  bucketSizesInput.buffer.destroy();
  bucketValuesInput.buffer.destroy();
  windowStartsInput.buffer.destroy();
  windowCountsInput.buffer.destroy();
  bucketOutput.destroy();
  bucketParams.buffer.destroy();
  windowOutput.destroy();
  windowParams.buffer.destroy();
  finalOutput.destroy();
  finalParams.buffer.destroy();

  profile.reductionTotalMs = profile.bucketReductionTotalMs + profile.windowReductionTotalMs + profile.finalReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return profile;
}

async function benchmarkMSM(
  device: GPUDevice,
  kernel: Kernel,
  bases: readonly JacobianPoint[],
  scalars: ScalarBatch,
  iters: number,
  run: () => Promise<MSMProfile>,
): Promise<{ cold: MSMProfile; warm: MSMProfile }> {
  const cold = await run();
  if (iters === 1) {
    return { cold, warm: cold };
  }
  const warm: MSMProfile = {
    partitionMs: 0,
    uploadMs: 0,
    kernelMs: 0,
    readbackMs: 0,
    scalarMulTotalMs: 0,
    bucketReductionTotalMs: 0,
    windowReductionTotalMs: 0,
    finalReductionTotalMs: 0,
    reductionTotalMs: 0,
    totalMs: 0,
  };
  for (let i = 0; i < iters; i += 1) {
    const profile = await run();
    warm.partitionMs += profile.partitionMs;
    warm.uploadMs += profile.uploadMs;
    warm.kernelMs += profile.kernelMs;
    warm.readbackMs += profile.readbackMs;
    warm.scalarMulTotalMs += profile.scalarMulTotalMs;
    warm.bucketReductionTotalMs += profile.bucketReductionTotalMs;
    warm.windowReductionTotalMs += profile.windowReductionTotalMs;
    warm.finalReductionTotalMs += profile.finalReductionTotalMs;
    warm.reductionTotalMs += profile.reductionTotalMs;
    warm.totalMs += profile.totalMs;
  }
  warm.partitionMs /= iters;
  warm.uploadMs /= iters;
  warm.kernelMs /= iters;
  warm.readbackMs /= iters;
  warm.scalarMulTotalMs /= iters;
  warm.bucketReductionTotalMs /= iters;
  warm.windowReductionTotalMs /= iters;
  warm.finalReductionTotalMs /= iters;
  warm.reductionTotalMs /= iters;
  warm.totalMs /= iters;
  return { cold, warm };
}

async function runBenchmark(): Promise<void> {
  const lines = ["=== BN254 G1 MSM Browser Benchmark ===", ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  mustElement(runButton, "run").disabled = true;

  try {
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
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    await appendAdapterDiagnostics(adapter, lines);
    lines.push("2. Requesting device... OK");
    const device = await adapter.requestDevice();

    const [shaderText, phase7] = await Promise.all([
      fetchText("/shaders/curves/bn254/g1_arith.wgsl"),
      fetchJSON<Phase7Vectors>("/testdata/vectors/g1/bn254_phase7_scalar_mul.json"),
    ]);
    lines.push("3. Loading shader and generator metadata... OK");

    const kernel = createKernel(device, shaderText);
    const pippengerKernels = {
      bucket: createKernel(device, shaderText, "g1_msm_bucket_sparse_main"),
      windowSparse: createKernel(device, shaderText, "g1_msm_window_sparse_main"),
      combine: createKernel(device, shaderText, "g1_msm_combine_main"),
    };
    const initMs = performance.now() - initStart;
    lines.push("4. Creating pipeline... OK");
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    lines.push("");
    lines.push("size,op,window,init_ms,cold_partition_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_scalar_mul_total_ms,cold_bucket_reduction_ms,cold_window_reduction_ms,cold_final_reduction_ms,cold_reduction_total_ms,cold_total_ms,cold_with_init_ms,warm_partition_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_scalar_mul_total_ms,warm_bucket_reduction_ms,warm_window_reduction_ms,warm_final_reduction_ms,warm_reduction_total_ms,warm_total_ms");

    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      const size = 1 << logSize;
      const bases = await buildBases(device, kernel, phase7.generator_affine, phase7.one_mont_z, size);
      const scalars = makeMSMScalars(size);
      const benchmarks = [
        {
          label: "msm_naive_affine",
          window: 0,
          run: () => runMSMProfiled(device, kernel, bases, scalars.hexes),
        },
        {
          label: "msm_pippenger_affine",
          window: bestPippengerWindow(size),
          run: () => runPippengerMSMProfiled(device, pippengerKernels, bases, scalars, bestPippengerWindow(size)),
        },
      ];
      for (const bench of benchmarks) {
        const benchmark = await benchmarkMSM(device, kernel, bases, scalars, iters, bench.run);
        lines.push(
          `${size},${bench.label},${bench.window},${initMs.toFixed(3)},${benchmark.cold.partitionMs.toFixed(3)},${benchmark.cold.uploadMs.toFixed(3)},${benchmark.cold.kernelMs.toFixed(3)},${benchmark.cold.readbackMs.toFixed(3)},${benchmark.cold.scalarMulTotalMs.toFixed(3)},${benchmark.cold.bucketReductionTotalMs.toFixed(3)},${benchmark.cold.windowReductionTotalMs.toFixed(3)},${benchmark.cold.finalReductionTotalMs.toFixed(3)},${benchmark.cold.reductionTotalMs.toFixed(3)},${benchmark.cold.totalMs.toFixed(3)},${(initMs + benchmark.cold.totalMs).toFixed(3)},${benchmark.warm.partitionMs.toFixed(3)},${benchmark.warm.uploadMs.toFixed(3)},${benchmark.warm.kernelMs.toFixed(3)},${benchmark.warm.readbackMs.toFixed(3)},${benchmark.warm.scalarMulTotalMs.toFixed(3)},${benchmark.warm.bucketReductionTotalMs.toFixed(3)},${benchmark.warm.windowReductionTotalMs.toFixed(3)},${benchmark.warm.finalReductionTotalMs.toFixed(3)},${benchmark.warm.reductionTotalMs.toFixed(3)},${benchmark.warm.totalMs.toFixed(3)}`,
        );
        writeLog(lines);
      }
    }

    lines.push("");
    lines.push("PASS: BN254 G1 MSM browser benchmark completed");
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
