export {};

import {
  bestPippengerWindow as sharedBestPippengerWindow,
  buildSparseSignedBucketMetadataWords,
  makeRandomScalarBatch,
} from "./curvegpu/msm_shared";
import {
  benchmarkMSM,
  formatMSMBenchmarkRow,
  MSMProfile,
} from "./curvegpu/msm_bench_shared";
import {
  createBindGroupForBuffers as sharedCreateBindGroupForBuffers,
  createEmptyPointStorageBuffer as sharedCreateEmptyPointStorageBuffer,
  createMSMKernel as sharedCreateMSMKernel,
  createParamsBuffer as sharedCreateParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer as sharedCreateU32StorageBuffer,
  Kernel,
  readbackBufferProfiled,
  submitKernelProfiled as sharedSubmitKernelProfiled,
} from "./curvegpu/msm_gpu_runtime";

type AffinePoint = {
  x_bytes_le: string;
  y_bytes_le: string;
};

type JacobianPoint = {
  x_bytes_le: string;
  y_bytes_le: string;
  z_bytes_le: string;
};

type FixtureMetadata = {
  count: number;
  point_bytes: number;
  format: string;
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

declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  MAP_READ: number;
  UNIFORM: number;
};
declare const GPUMapMode: { READ: number };

const FP_BYTES = 48;
const POINT_BYTES = 144;
const UNIFORM_BYTES = 32;
const ZERO_HEX =
  "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
const INDEX_SIGN_BIT = 0x80000000;
const G1_OP_JAC_INFINITY = 1;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;

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

async function fetchBytes(path: string): Promise<Uint8Array> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
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
  return sharedCreateMSMKernel(device, shaderCode, "bls12-381-g1", entryPoint);
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

function jacToAffinePoint(point: JacobianPoint, oneMontZ: string): JacobianPoint {
  if (point.z_bytes_le === ZERO_HEX) {
    return zeroPoint();
  }
  return { x_bytes_le: point.x_bytes_le, y_bytes_le: point.y_bytes_le, z_bytes_le: oneMontZ };
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
  return sharedBestPippengerWindow(count);
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
  return buildSparseSignedBucketMetadataWords(scalarWords, count, termsPerInstance, window, maxChunkSize);
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

function makeScalarHexLEFromUint64(value: bigint): string {
  const out = new Uint8Array(32);
  let x = value;
  for (let i = 0; i < 8; i += 1) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return bytesToHex(out);
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

async function runScalarMulAffineProfiled(
  device: GPUDevice,
  kernel: Kernel,
  base: readonly JacobianPoint[],
  scalarsHexLE: readonly string[],
): Promise<{ out: JacobianPoint[]; profile: MSMProfile }> {
  const totalStart = performance.now();
  let uploadMs = 0;
  let kernelMs = 0;
  let readbackMs = 0;
  let acc = await runOpProfiled(device, kernel, G1_OP_JAC_INFINITY, base.map(() => zeroPoint()), base.map(() => zeroPoint()));
  uploadMs += acc.profile.uploadMs;
  kernelMs += acc.profile.kernelMs;
  readbackMs += acc.profile.readbackMs;
  for (let bit = 255; bit >= 0; bit -= 1) {
    const doubled = await runOpProfiled(device, kernel, G1_OP_DOUBLE_JAC, acc.out, acc.out.map(() => zeroPoint()));
    uploadMs += doubled.profile.uploadMs;
    kernelMs += doubled.profile.kernelMs;
    readbackMs += doubled.profile.readbackMs;
    acc = doubled;
    const mask = scalarsHexLE.map((scalar) => scalarBit(scalar, bit));
    if (!anyMask(mask)) {
      continue;
    }
    const added = await runOpProfiled(device, kernel, G1_OP_ADD_MIXED, acc.out, maskedAffine(base, mask));
    uploadMs += added.profile.uploadMs;
    kernelMs += added.profile.kernelMs;
    readbackMs += added.profile.readbackMs;
    acc = added;
  }
  const affine = await runOpProfiled(device, kernel, G1_OP_JAC_TO_AFFINE, acc.out, acc.out.map(() => zeroPoint()));
  uploadMs += affine.profile.uploadMs;
  kernelMs += affine.profile.kernelMs;
  readbackMs += affine.profile.readbackMs;
  return {
    out: affine.out,
    profile: {
      partitionMs: 0,
      uploadMs,
      kernelMs,
      readbackMs,
      scalarMulTotalMs: performance.now() - totalStart,
      bucketReductionTotalMs: 0,
      windowReductionTotalMs: 0,
      finalReductionTotalMs: 0,
      reductionTotalMs: 0,
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
  return createStorageBufferFromBytes(device, label, bytes, count * POINT_BYTES);
}

function createPointStorageBufferFromBytes(
  device: GPUDevice,
  label: string,
  bytes: Uint8Array,
  count: number,
): { buffer: GPUBuffer; uploadMs: number } {
  return createStorageBufferFromBytes(device, label, bytes, Math.max(1, count) * POINT_BYTES);
}

function createU32StorageBuffer(
  device: GPUDevice,
  label: string,
  values: Uint32Array,
): { buffer: GPUBuffer; uploadMs: number } {
  return sharedCreateU32StorageBuffer(device, label, values);
}

function createEmptyPointStorageBuffer(device: GPUDevice, label: string, count: number): GPUBuffer {
  return sharedCreateEmptyPointStorageBuffer(device, label, count, POINT_BYTES);
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
  return sharedCreateParamsBuffer(device, label, UNIFORM_BYTES, values);
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
  return sharedCreateBindGroupForBuffers(device, kernel, label, inputA, inputB, output, params, meta0, meta1, meta2);
}

async function submitKernelProfiled(
  device: GPUDevice,
  kernel: Kernel,
  bindGroup: GPUBindGroup,
  count: number,
  label: string,
): Promise<number> {
  return sharedSubmitKernelProfiled(device, kernel, bindGroup, count, label);
}

async function readbackPointBufferProfiled(
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<{ points: JacobianPoint[]; readbackMs: number }> {
  const result = await readbackBufferProfiled(device, buffer, Math.max(1, count) * POINT_BYTES);
  return { points: unpackPointBatch(result.bytes, count), readbackMs: result.readbackMs };
}

function sliceBaseFixture(fixture: Uint8Array, count: number): Uint8Array {
  const byteLength = count * POINT_BYTES;
  if (byteLength > fixture.byteLength) {
    throw new Error(`fixture has ${Math.floor(fixture.byteLength / POINT_BYTES)} points, need ${count}`);
  }
  return fixture.slice(0, byteLength);
}

function makeMSMScalars(count: number): ScalarBatch {
  return makeRandomScalarBatch(count);
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

  const scalarMul = await runScalarMulAffineProfiled(device, kernel, bases, scalars);
  uploadMs += scalarMul.profile.uploadMs;
  kernelMs += scalarMul.profile.kernelMs;
  readbackMs += scalarMul.profile.readbackMs;
  scalarMulTotalMs += scalarMul.profile.totalMs;

  let width = bases.length;
  let state = scalarMul.out;
  while (width > 1) {
    const nextWidth = Math.ceil(width / 2);
    const left: JacobianPoint[] = [];
    const right: JacobianPoint[] = [];
    for (let j = 0; j < nextWidth; j += 1) {
      left.push(jacToAffinePoint(state[2 * j], bases[0]?.z_bytes_le ?? ZERO_HEX));
      right.push(2 * j + 1 < width ? jacToAffinePoint(state[2 * j + 1], bases[0]?.z_bytes_le ?? ZERO_HEX) : zeroPoint());
    }
    const stage = await runOpProfiled(device, kernel, G1_OP_AFFINE_ADD, left, right);
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
    weightBuckets: Kernel;
    subsumPhase1: Kernel;
    combine: Kernel;
  },
  basesBytes: Uint8Array,
  scalars: ScalarBatch,
  window: number,
): Promise<MSMProfile> {
  const totalStart = performance.now();
  const count = 1;
  const termsPerInstance = Math.floor(basesBytes.byteLength / POINT_BYTES);
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
  const basesInput = createPointStorageBufferFromBytes(device, "g1-pip-bases", basesBytes, termsPerInstance);
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

  const bucketValuesInput = createU32StorageBuffer(device, "g1-pip-window-values", metadata.bucketValues);
  profile.uploadMs += bucketValuesInput.uploadMs;
  const weightedBucketOutput = createEmptyPointStorageBuffer(device, "g1-pip-weighted-bucket-out", bucketCountOut);
  const weightParams = createParamsBuffer(device, "g1-pip-weight-params", {
    count: bucketCountOut,
  });
  profile.uploadMs += weightParams.uploadMs;
  const weightBindGroup = createBindGroupForBuffers(
    device,
    kernels.weightBuckets,
    "g1-pip-weight-bg",
    bucketOutput,
    zeroInput.buffer,
    weightedBucketOutput,
    weightParams.buffer,
    bucketValuesInput.buffer,
  );
  const weightStart = performance.now();
  const weightKernelMs = await submitKernelProfiled(device, kernels.weightBuckets, weightBindGroup, bucketCountOut, "g1-pip-weight");
  profile.kernelMs += weightKernelMs;
  profile.windowReductionTotalMs += performance.now() - weightStart;

  const windowOutput = createEmptyPointStorageBuffer(device, "g1-pip-window-sparse-out", count * metadata.numWindows);
  const windowStartsInput = createU32StorageBuffer(device, "g1-pip-window-starts", metadata.windowStarts);
  const windowCountsInput = createU32StorageBuffer(device, "g1-pip-window-counts", metadata.windowCounts);
  const windowParams = createParamsBuffer(device, "g1-pip-window-sparse-params", {
    count: count * metadata.numWindows,
  });
  profile.uploadMs += windowStartsInput.uploadMs + windowCountsInput.uploadMs + windowParams.uploadMs;
  const windowBindGroup = createBindGroupForBuffers(
    device,
    kernels.subsumPhase1,
    "g1-pip-window-sparse-bg",
    weightedBucketOutput,
    zeroInput.buffer,
    windowOutput,
    windowParams.buffer,
    bucketValuesInput.buffer,
    windowStartsInput.buffer,
    windowCountsInput.buffer,
  );
  const windowStart = performance.now();
  const windowKernelMs = await submitKernelProfiled(device, kernels.subsumPhase1, windowBindGroup, count * metadata.numWindows * 64, "g1-pip-window-sparse");
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
  weightedBucketOutput.destroy();
  bucketParams.buffer.destroy();
  weightParams.buffer.destroy();
  windowOutput.destroy();
  windowParams.buffer.destroy();
  finalOutput.destroy();
  finalParams.buffer.destroy();

  profile.reductionTotalMs = profile.bucketReductionTotalMs + profile.windowReductionTotalMs + profile.finalReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return profile;
}

async function runBenchmark(): Promise<void> {
  const lines = ["=== BLS12-381 G1 MSM Browser Benchmark ===", ""];
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

    const [shaderText, fixtureMeta] = await Promise.all([
      fetchText("/shaders/curves/bls12_381/g1_msm.wgsl?v=1"),
      fetchJSON<FixtureMetadata>("/testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.json?v=1"),
    ]);
    const fixtureLoadStart = performance.now();
    const baseFixture = await fetchBytes("/testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.bin?v=1");
    const fixtureLoadMs = performance.now() - fixtureLoadStart;
    if (fixtureMeta.point_bytes !== POINT_BYTES) {
      throw new Error(`unexpected fixture point size: ${fixtureMeta.point_bytes}`);
    }
    if (baseFixture.byteLength !== fixtureMeta.count * fixtureMeta.point_bytes) {
      throw new Error(`fixture length mismatch: got ${baseFixture.byteLength}, want ${fixtureMeta.count * fixtureMeta.point_bytes}`);
    }
    lines.push("3. Loading shader and base fixture... OK");

    const kernel = createKernel(device, shaderText);
    const pippengerKernels = {
      bucket: createKernel(device, shaderText, "g1_msm_bucket_sparse_main"),
      weightBuckets: createKernel(device, shaderText, "g1_msm_weight_buckets_main"),
      subsumPhase1: createKernel(device, shaderText, "g1_msm_subsum_phase1_main"),
      combine: createKernel(device, shaderText, "g1_msm_combine_main"),
    };
    const initMs = performance.now() - initStart;
    lines.push("4. Creating pipeline... OK");
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    lines.push(`fixture_load_ms = ${fixtureLoadMs.toFixed(3)}`);
    lines.push("");
    lines.push("size,op,window,init_ms,prep_ms,cold_partition_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_scalar_mul_total_ms,cold_bucket_reduction_ms,cold_window_reduction_ms,cold_final_reduction_ms,cold_reduction_total_ms,cold_total_ms,cold_with_init_prep_ms,warm_partition_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_scalar_mul_total_ms,warm_bucket_reduction_ms,warm_window_reduction_ms,warm_final_reduction_ms,warm_reduction_total_ms,warm_total_ms");

    for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
      const size = 1 << logSize;
      const prepStart = performance.now();
      const bases = sliceBaseFixture(baseFixture, size);
      const prepMs = performance.now() - prepStart;
      const scalars = makeMSMScalars(size);
      const benchmarks = [
        {
          label: "msm_pippenger_affine",
          window: bestPippengerWindow(size),
          run: () => runPippengerMSMProfiled(device, pippengerKernels, bases, scalars, bestPippengerWindow(size)),
        },
      ];
      for (const bench of benchmarks) {
        const benchmark = await benchmarkMSM(iters, bench.run);
        lines.push(formatMSMBenchmarkRow({
          size,
          label: bench.label,
          window: bench.window,
          initMs,
          prepMs,
          includePrepMs: true,
          cold: benchmark.cold,
          warm: benchmark.warm,
        }));
        writeLog(lines);
      }
    }

    lines.push("");
    lines.push("PASS: BLS12-381 G1 MSM browser benchmark completed");
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
