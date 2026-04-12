export {};

import {
  appendAdapterDiagnostics,
  bytesToHex,
  createPageUI,
  fetchBytes,
  fetchJSON,
  fetchText,
  hexToBytes,
  mustElement,
} from "./curvegpu/browser_utils.js";
import { fetchShaderParts } from "./curvegpu/shaders.js";

import {
  bestPippengerWindow as sharedBestPippengerWindow,
  buildSparseSignedBucketMetadataWords,
  makeRandomScalarBatch,
} from "./curvegpu/msm_shared.js";
import {
  appendMSMBenchmarkRows,
  benchmarkMSM,
  MSMProfile,
} from "./curvegpu/msm_bench_shared.js";
import { runMSMBenchmarkPage } from "./curvegpu/msm_page_runner.js";
import { runSparseSignedPippengerMSMProfiled, type PippengerBenchKernels } from "./curvegpu/msm_pippenger_bench.js";
import { createPreferredByteBaseSource } from "./curvegpu/msm_bench_sources.js";
import {
  createBindGroupForBuffers as sharedCreateBindGroupForBuffers,
  createEmptyPointStorageBuffer as sharedCreateEmptyPointStorageBuffer,
  createMSMKernel as sharedCreateMSMKernel,
  createMSMKernelSet,
  createParamsBuffer as sharedCreateParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer as sharedCreateU32StorageBuffer,
  Kernel,
  readbackBufferProfiled,
  submitKernelProfiled as sharedSubmitKernelProfiled,
} from "./curvegpu/msm_gpu_runtime.js";

type AffinePoint = {
  x_bytes_le: string;
  y_bytes_le: string;
};

type JacobianPoint = {
  x_bytes_le: string;
  y_bytes_le: string;
  z_bytes_le: string;
};

type G1ScalarMulVectors = {
  generator_affine: AffinePoint;
  one_mont_z: string;
};

type CurveId = "bn254" | "bls12_381";

type CurveBenchConfig = {
  id: CurveId;
  title: string;
  successMessage: string;
  opsKernelLabel: string;
  pippengerKernelLabel: string;
  fpBytes: number;
  pointBytes: number;
  zeroHex: string;
  opsShaderParts: string[];
  pippengerShaderParts: string[];
  scalarVectorsPath: string;
  fixtureJSONPath?: string;
  fixtureBinPath?: string;
  serverBinPath?: string;
  pippengerMode: "simple" | "weighted";
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

const UNIFORM_BYTES = 32;
const INDEX_SIGN_BIT = 0x80000000;
const G1_OP_JAC_INFINITY = 1;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;
const G1_OP_SCALAR_MUL_AFFINE = 8;

const CURVE_CONFIGS: Record<CurveId, CurveBenchConfig> = {
  bn254: {
    id: "bn254",
    title: "BN254 G1 MSM Browser Benchmark",
    successMessage: "BN254 G1 MSM browser benchmark completed",
    opsKernelLabel: "bn254-g1",
    pippengerKernelLabel: "bn254-g1",
    fpBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
    opsShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl?v=2#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl?v=1",
    ],
    pippengerShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl?v=2#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl?v=1",
    ],
    scalarVectorsPath: "/testdata/vectors/g1/bn254_g1_scalar_mul.json",
    serverBinPath: "/api/bn254/g1/bases.bin",
    pippengerMode: "simple",
  },
  bls12_381: {
    id: "bls12_381",
    title: "BLS12-381 G1 MSM Browser Benchmark",
    successMessage: "BLS12-381 G1 MSM browser benchmark completed",
    opsKernelLabel: "bls12-381-g1",
    pippengerKernelLabel: "bls12-381-g1",
    fpBytes: 48,
    pointBytes: 144,
    zeroHex:
      "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    opsShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=3#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-core",
      "/shaders/curves/bls12_381/g1_arith.wgsl?v=2",
    ],
    pippengerShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=3#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-core",
      "/shaders/curves/bls12_381/g1_msm.wgsl?v=3",
    ],
    scalarVectorsPath: "/testdata/vectors/g1/bls12_381_g1_scalar_mul.json",
    fixtureJSONPath: "/testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.json?v=1",
    fixtureBinPath: "/testdata/fixtures/g1/bls12_381_bases_2pow19_jacobian.bin?v=1",
    serverBinPath: "/api/bls12-381/g1/bases.bin",
    pippengerMode: "weighted",
  },
};

function getCurveConfig(): CurveBenchConfig {
  const params = new URLSearchParams(window.location.search);
  const curve = (params.get("curve") ?? "bn254") as CurveId;
  return CURVE_CONFIGS[curve] ?? CURVE_CONFIGS.bn254;
}

const curveConfig = getCurveConfig();
const FP_BYTES = curveConfig.fpBytes;
const POINT_BYTES = curveConfig.pointBytes;
const ZERO_HEX = curveConfig.zeroHex;

const minLogEl = document.getElementById("min-log") as HTMLInputElement | null;
const maxLogEl = document.getElementById("max-log") as HTMLInputElement | null;
const itersEl = document.getElementById("iters") as HTMLInputElement | null;
const runButton = document.getElementById("run") as HTMLButtonElement | null;
const statusEl = document.getElementById("status") as HTMLElement | null;
const logEl = document.getElementById("log") as HTMLElement | null;
const { setStatus, setPageState, writeLog } = createPageUI(statusEl, logEl);

function createKernel(device: GPUDevice, shaderCode: string, entryPoint = "g1_ops_main"): Kernel {
  return sharedCreateMSMKernel(device, shaderCode, curveConfig.opsKernelLabel, entryPoint);
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

function makeMSMScalars(count: number): ScalarBatch {
  return makeRandomScalarBatch(count);
}

async function buildBases(
  device: GPUDevice,
  kernel: Kernel,
  generator: AffinePoint,
  oneMontZ: string,
  count: number,
): Promise<Uint8Array> {
  const generatorKernel = affineToKernelPoint(generator, oneMontZ);
  const bases = Array.from({ length: count }, () => generatorKernel);
  const scalars = Array.from({ length: count }, (_, index) => makeScalarHexLEFromUint64(BigInt(index + 1)));
  const generated = await runScalarMulAffineProfiled(device, kernel, bases, scalars);
  return packPointBatch(generated.out);
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
  kernels: PippengerBenchKernels,
  basesBytes: Uint8Array,
  scalars: ScalarBatch,
  window: number,
): Promise<MSMProfile> {
  return runSparseSignedPippengerMSMProfiled({
    device,
    kernels,
    basesBytes,
    pointBytes: POINT_BYTES,
    uniformBytes: UNIFORM_BYTES,
    zeroPointBytes: packPointBatch([zeroPoint()]),
    scalarWords: scalars.words,
    count: 1,
    termsPerInstance: Math.floor(basesBytes.byteLength / POINT_BYTES),
    window,
    labelPrefix: "g1-pip",
  });
}

async function runBenchmark(): Promise<void> {
  const tableHeader =
    "size,op,window,init_ms,prep_ms,cold_partition_ms,cold_upload_ms,cold_kernel_ms,cold_readback_ms,cold_scalar_mul_total_ms,cold_bucket_reduction_ms,cold_window_reduction_ms,cold_final_reduction_ms,cold_reduction_total_ms,cold_total_ms,cold_with_init_prep_ms,warm_partition_ms,warm_upload_ms,warm_kernel_ms,warm_readback_ms,warm_scalar_mul_total_ms,warm_bucket_reduction_ms,warm_window_reduction_ms,warm_final_reduction_ms,warm_reduction_total_ms,warm_total_ms";
  await runMSMBenchmarkPage({
    title: curveConfig.title,
    successMessage: curveConfig.successMessage,
    tableHeader,
    elements: {
      minLogEl: mustElement(minLogEl, "min-log"),
      maxLogEl: mustElement(maxLogEl, "max-log"),
      itersEl: mustElement(itersEl, "iters"),
      runButton: mustElement(runButton, "run"),
    },
    ui: { setStatus, setPageState, writeLog },
    appendAdapterDiagnostics,
    init: async ({ device, lines }) => {
      const [opsShaderText, pippengerShaderText, scalarVectors] = await Promise.all([
        fetchShaderParts(curveConfig.opsShaderParts),
        fetchShaderParts(curveConfig.pippengerShaderParts),
        fetchJSON<G1ScalarMulVectors>(curveConfig.scalarVectorsPath),
      ]);
      const opsKernel = createKernel(device, opsShaderText);
      const baseSourceProvider = createPreferredByteBaseSource({
        locationSearch: window.location.search,
        pointBytes: POINT_BYTES,
        fixtureJSONPath: curveConfig.fixtureJSONPath,
        fixtureBinPath: curveConfig.fixtureBinPath,
        serverBinPath: curveConfig.serverBinPath,
        generatedLoadBases: async (size) =>
          buildBases(device, opsKernel, scalarVectors.generator_affine, scalarVectors.one_mont_z, size),
      });
      const baseSourceInit = await baseSourceProvider.init();
      lines.push(`3. Loading shader and base source... OK (${baseSourceInit.context.baseSource})`);
      const pippengerKernels: PippengerBenchKernels =
        curveConfig.pippengerMode === "weighted"
          ? {
              mode: "weighted",
              ...createMSMKernelSet(device, pippengerShaderText, curveConfig.pippengerKernelLabel, {
                bucket: "g1_msm_bucket_sparse_main",
                weightBuckets: "g1_msm_weight_buckets_main",
                subsumPhase1: "g1_msm_subsum_phase1_main",
                combine: "g1_msm_combine_main",
              }),
            }
          : {
              mode: "simple",
              ...createMSMKernelSet(device, pippengerShaderText, curveConfig.pippengerKernelLabel, {
                bucket: "g1_msm_bucket_sparse_main",
                windowSparse: "g1_msm_window_sparse_main",
                combine: "g1_msm_combine_main",
              }),
            };
      return {
        context: { device, pippengerKernels, baseSourceProvider, baseSourceContext: baseSourceInit.context },
        preMetricLines: ["4. Creating pipeline... OK"],
        postMetricLines: baseSourceInit.postMetricLines,
      };
    },
    runSizes: async ({ context, lines, initMs, minLog, maxLog, iters, writeLog }) => {
      await appendMSMBenchmarkRows({
        lines,
        initMs,
        minLog,
        maxLog,
        iters,
        writeLog,
        makeSizeBenchmarks: async ({ size }) => {
          const { bases: basesBytes, prepMs } = await context.baseSourceProvider.loadBases({
            context: context.baseSourceContext,
            size,
          });
          const scalars = makeMSMScalars(size);
          const bases = unpackPointBatch(basesBytes, size);
          const window = bestPippengerWindow(size);
          return {
            prepMs,
            includePrepMs: true,
            entries: [
              {
                label: "msm_pippenger_affine",
                window,
                run: () =>
                  runPippengerMSMProfiled(
                    context.device,
                    context.pippengerKernels,
                    basesBytes,
                    scalars,
                    window,
                  ),
              },
            ],
          };
        },
      });
    },
  });
}

mustElement(runButton, "run").addEventListener("click", () => {
  void runBenchmark();
});
