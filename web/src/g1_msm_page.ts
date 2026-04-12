export {};

import { fetchShaderParts } from "./curvegpu/shaders.js";

import {
  bestPippengerWindow as sharedBestPippengerWindow,
  buildSparseSignedBucketMetadataWords,
  hexesToScalarWords,
} from "./curvegpu/msm_shared.js";
import {
  createBindGroupForBuffers as sharedCreateBindGroupForBuffers,
  createEmptyPointStorageBuffer as sharedCreateEmptyPointStorageBuffer,
  createMSMKernel as sharedCreateMSMKernel,
  createParamsBuffer as sharedCreateParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer as sharedCreateU32StorageBuffer,
  Kernel as SharedMSMKernel,
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

type CurveId = "bn254" | "bls12_381";

type CurveConfig = {
  id: CurveId;
  logTitle: string;
  passLine: string;
  opsKernelLabel: string;
  msmKernelLabel: string;
  fpBytes: number;
  pointBytes: number;
  zeroHex: string;
  vectorsPath: string;
  opsShaderParts: string[];
  msmShaderParts?: string[];
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
const G1_OP_SCALAR_MUL_AFFINE = 8;

const UNIFORM_BYTES = 32;
const INDEX_SIGN_BIT = 0x80000000;

const CURVE_CONFIGS: Record<CurveId, CurveConfig> = {
  bn254: {
    id: "bn254",
    logTitle: "=== BN254 G1 Phase 8 Browser Smoke ===",
    passLine: "PASS: BN254 G1 Phase 8 browser smoke succeeded",
    opsKernelLabel: "bn254-g1",
    msmKernelLabel: "bn254-g1-msm",
    fpBytes: 32,
    pointBytes: 96,
    zeroHex: "0000000000000000000000000000000000000000000000000000000000000000",
    vectorsPath: "/testdata/vectors/g1/bn254_phase8_msm.json",
    opsShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl?v=2#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl?v=1",
    ],
    msmShaderParts: [
      "/shaders/curves/bn254/fp_arith.wgsl?v=2#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl?v=3#section=fp-core",
      "/shaders/curves/bn254/g1_arith.wgsl?v=1",
    ],
  },
  bls12_381: {
    id: "bls12_381",
    logTitle: "=== BLS12-381 G1 Phase 8 Browser Smoke ===",
    passLine: "PASS: BLS12-381 G1 Phase 8 browser smoke succeeded",
    opsKernelLabel: "bls12-381-g1",
    msmKernelLabel: "bls12-381-g1-msm",
    fpBytes: 48,
    pointBytes: 144,
    zeroHex:
      "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    vectorsPath: "/testdata/vectors/g1/bls12_381_phase8_msm.json?v=1",
    opsShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=3#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-core",
      "/shaders/curves/bls12_381/g1_arith.wgsl?v=2",
    ],
    msmShaderParts: [
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=3#section=fp-types",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-consts",
      "/shaders/curves/bls12_381/fp_arith.wgsl?v=4#section=fp-core",
      "/shaders/curves/bls12_381/g1_msm.wgsl?v=3",
    ],
  },
};

function getCurveConfig(): CurveConfig {
  const params = new URLSearchParams(window.location.search);
  const curve = (params.get("curve") ?? "bn254") as CurveId;
  return CURVE_CONFIGS[curve] ?? CURVE_CONFIGS.bn254;
}

const curveConfig = getCurveConfig();
const FP_BYTES = curveConfig.fpBytes;
const POINT_BYTES = curveConfig.pointBytes;
const ZERO_HEX = curveConfig.zeroHex;

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
  const text = await fetchText(curveConfig.vectorsPath);
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
    label: `${curveConfig.opsKernelLabel}-shader`,
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${curveConfig.opsKernelLabel}-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${curveConfig.opsKernelLabel}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${curveConfig.opsKernelLabel}-pipeline`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "g1_ops_main" },
  });
  return { pipeline, bindGroupLayout };
}

function createMSMKernel(device: GPUDevice, shaderCode: string, entryPoint: string): SharedMSMKernel {
  return sharedCreateMSMKernel(device, shaderCode, curveConfig.msmKernelLabel, entryPoint);
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
      throw new Error(
        `${name}: mismatch at index ${i}` +
          ` got=(${got[i].x_bytes_le},${got[i].y_bytes_le},${got[i].z_bytes_le})` +
          ` want=(${want[i].x_bytes_le},${want[i].y_bytes_le},${want[i].z_bytes_le})`,
      );
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
  return sharedBestPippengerWindow(count);
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
  return buildSparseSignedBucketMetadataWords(hexesToScalarWords(scalars), count, termsPerInstance, window, maxChunkSize);
}

async function runScalarMulAffineBLS(
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

async function runNaiveMSMBLS(
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

  const terms = await runScalarMulAffineBLS(device, kernel, bases, scalars);
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

function createPointStorageBufferBLS(device: GPUDevice, label: string, points: readonly JacobianPoint[], minCount = points.length): GPUBuffer {
  const bytes = packPointBatch(points.length === 0 ? [zeroPoint()] : points);
  const count = Math.max(minCount, points.length === 0 ? 1 : points.length);
  return createStorageBufferFromBytes(device, label, bytes, count * POINT_BYTES).buffer;
}

function createU32StorageBufferBLS(device: GPUDevice, label: string, values: Uint32Array): GPUBuffer {
  return sharedCreateU32StorageBuffer(device, label, values).buffer;
}

function createEmptyPointStorageBufferBLS(device: GPUDevice, label: string, count: number): GPUBuffer {
  return sharedCreateEmptyPointStorageBuffer(device, label, count, POINT_BYTES);
}

function createParamsBufferBLS(
  device: GPUDevice,
  label: string,
  values: { count: number; termsPerInstance?: number; window?: number; numWindows?: number; bucketCount?: number; rowWidth?: number },
): GPUBuffer {
  return sharedCreateParamsBuffer(device, label, UNIFORM_BYTES, values).buffer;
}

function createBindGroupForBuffersBLS(
  device: GPUDevice,
  kernel: SharedMSMKernel,
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

async function submitKernelBLS(device: GPUDevice, kernel: SharedMSMKernel, bindGroup: GPUBindGroup, count: number): Promise<void> {
  await sharedSubmitKernelProfiled(device, kernel, bindGroup, count, curveConfig.msmKernelLabel);
}

async function readbackPointBufferBLS(device: GPUDevice, buffer: GPUBuffer, count: number): Promise<JacobianPoint[]> {
  const result = await readbackBufferProfiled(device, buffer, Math.max(1, count) * POINT_BYTES);
  return unpackPointBatch(result.bytes, count);
}

async function runOptimizedPippengerMSMBLS(
  device: GPUDevice,
  kernels: { bucket: SharedMSMKernel; weightBuckets: SharedMSMKernel; subsumPhase1: SharedMSMKernel; combine: SharedMSMKernel },
  bases: readonly JacobianPoint[],
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
  window: number,
): Promise<JacobianPoint[]> {
  const metadata = buildSparseSignedBucketMetadata(scalars, count, termsPerInstance, window);
  const zeroBuffer = createPointStorageBufferBLS(device, "g1-msm-zero", [zeroPoint()], 1);
  const basesBuffer = createPointStorageBufferBLS(device, "g1-msm-bases", bases);
  const baseIndicesBuffer = createU32StorageBufferBLS(device, "g1-msm-base-indices", metadata.baseIndices);
  const bucketPointersBuffer = createU32StorageBufferBLS(device, "g1-msm-bucket-pointers", metadata.bucketPointers);
  const bucketSizesBuffer = createU32StorageBufferBLS(device, "g1-msm-bucket-sizes", metadata.bucketSizes);
  const bucketValuesBuffer = createU32StorageBufferBLS(device, "g1-msm-bucket-values", metadata.bucketValues);
  const windowStartsBuffer = createU32StorageBufferBLS(device, "g1-msm-window-starts", metadata.windowStarts);
  const windowCountsBuffer = createU32StorageBufferBLS(device, "g1-msm-window-counts", metadata.windowCounts);

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBufferBLS(device, "g1-msm-bucket-out", bucketCountOut);
  const bucketParams = createParamsBufferBLS(device, "g1-msm-bucket-params", {
    count: bucketCountOut,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  await submitKernelBLS(
    device,
    kernels.bucket,
    createBindGroupForBuffersBLS(
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

  const weightedBucketOutput = createEmptyPointStorageBufferBLS(device, "g1-msm-weighted-out", bucketCountOut);
  const weightParams = createParamsBufferBLS(device, "g1-msm-weight-params", { count: bucketCountOut });
  await submitKernelBLS(
    device,
    kernels.weightBuckets,
    createBindGroupForBuffersBLS(
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

  const windowOutput = createEmptyPointStorageBufferBLS(device, "g1-msm-window-out", count * metadata.numWindows);
  const windowParams = createParamsBufferBLS(device, "g1-msm-window-params", { count: count * metadata.numWindows });
  await submitKernelBLS(
    device,
    kernels.subsumPhase1,
    createBindGroupForBuffersBLS(
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

  const finalOutput = createEmptyPointStorageBufferBLS(device, "g1-msm-final-out", count);
  const finalParams = createParamsBufferBLS(device, "g1-msm-final-params", {
    count,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  await submitKernelBLS(
    device,
    kernels.combine,
    createBindGroupForBuffersBLS(
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

  const result = await readbackPointBufferBLS(device, finalOutput, count);

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

function scalarToKernelPointBN254(scalar: string): JacobianPoint {
  return { x_bytes_le: scalar, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX };
}

function createKernelBN254(device: GPUDevice, shaderCode: string, entryPoint = "g1_ops_main"): {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
} {
  const shaderModule = device.createShaderModule({
    label: `${curveConfig.opsKernelLabel}-shader`,
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${curveConfig.opsKernelLabel}-bgl`,
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
    label: `${curveConfig.opsKernelLabel}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${curveConfig.opsKernelLabel}-${entryPoint}`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

async function runOpBN254(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
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
): Promise<JacobianPoint[]> {
  const count = extraParams?.count ?? inputA.length;
  if (count === 0) {
    return [];
  }
  const aBytes = packPointBatch(inputA);
  const bBytes = packPointBatch(inputB.length === 0 ? [zeroPoint()] : inputB);
  const inputCount = Math.max(inputA.length, inputB.length === 0 ? 1 : inputB.length, count);
  const inputByteSize = inputCount * POINT_BYTES;
  const outputByteSize = count * POINT_BYTES;

  const inputABuffer = device.createBuffer({ label: "g1-input-a", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const inputBBuffer = device.createBuffer({ label: "g1-input-b", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ label: "g1-output", size: outputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ label: "g1-staging", size: outputByteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniformBuffer = device.createBuffer({ label: "g1-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const metaDummyBuffer = device.createBuffer({ label: "g1-meta-dummy", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(inputABuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  device.queue.writeBuffer(inputBBuffer, 0, bBytes.buffer.slice(bBytes.byteOffset, bBytes.byteOffset + bBytes.byteLength));
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
      { binding: 0, resource: { buffer: inputABuffer } },
      { binding: 1, resource: { buffer: inputBBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: metaDummyBuffer } },
      { binding: 5, resource: { buffer: metaDummyBuffer } },
      { binding: 6, resource: { buffer: metaDummyBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "g1-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  inputABuffer.destroy();
  inputBBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  metaDummyBuffer.destroy();

  return unpackPointBatch(result, count);
}

async function runNaiveMSMBN254(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  bases: readonly JacobianPoint[],
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
): Promise<JacobianPoint[]> {
  let state = await runOpBN254(device, kernel, G1_OP_SCALAR_MUL_AFFINE, bases, scalars.map((scalar) => scalarToKernelPointBN254(scalar)));
  let width = termsPerInstance;
  while (width > 1) {
    const nextWidth = Math.ceil(width / 2);
    const left: JacobianPoint[] = [];
    const right: JacobianPoint[] = [];
    for (let instance = 0; instance < count; instance += 1) {
      const rowBase = instance * width;
      for (let j = 0; j < nextWidth; j += 1) {
        left.push(state[rowBase + 2 * j]);
        right.push(2 * j + 1 < width ? state[rowBase + 2 * j + 1] : zeroPoint());
      }
    }
    state = await runOpBN254(device, kernel, G1_OP_AFFINE_ADD, left, right);
    width = nextWidth;
  }
  return state;
}

async function runSparseBucketOpBN254(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  bases: readonly JacobianPoint[],
  metadata: { baseIndices: Uint32Array; bucketPointers: Uint32Array; bucketSizes: Uint32Array },
  count: number,
): Promise<JacobianPoint[]> {
  if (count === 0) {
    return [];
  }
  const aBytes = packPointBatch(bases);
  const outputByteSize = count * POINT_BYTES;
  const basesBuffer = device.createBuffer({ label: "g1-sparse-bases", size: aBytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const dummyBuffer = device.createBuffer({ label: "g1-sparse-dummy", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ label: "g1-sparse-output", size: outputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ label: "g1-sparse-staging", size: outputByteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniformBuffer = device.createBuffer({ label: "g1-sparse-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const baseIndicesBuffer = device.createBuffer({ label: "g1-sparse-base-indices", size: Math.max(4, metadata.baseIndices.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bucketPointersBuffer = device.createBuffer({ label: "g1-sparse-bucket-pointers", size: Math.max(4, metadata.bucketPointers.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bucketSizesBuffer = device.createBuffer({ label: "g1-sparse-bucket-sizes", size: Math.max(4, metadata.bucketSizes.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(basesBuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  if (metadata.baseIndices.byteLength > 0) {
    device.queue.writeBuffer(baseIndicesBuffer, 0, metadata.baseIndices.buffer.slice(metadata.baseIndices.byteOffset, metadata.baseIndices.byteOffset + metadata.baseIndices.byteLength));
  }
  if (metadata.bucketPointers.byteLength > 0) {
    device.queue.writeBuffer(bucketPointersBuffer, 0, metadata.bucketPointers.buffer.slice(metadata.bucketPointers.byteOffset, metadata.bucketPointers.byteOffset + metadata.bucketPointers.byteLength));
  }
  if (metadata.bucketSizes.byteLength > 0) {
    device.queue.writeBuffer(bucketSizesBuffer, 0, metadata.bucketSizes.buffer.slice(metadata.bucketSizes.byteOffset, metadata.bucketSizes.byteOffset + metadata.bucketSizes.byteLength));
  }
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    label: "g1-sparse-bucket-bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: basesBuffer } },
      { binding: 1, resource: { buffer: dummyBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: baseIndicesBuffer } },
      { binding: 5, resource: { buffer: bucketPointersBuffer } },
      { binding: 6, resource: { buffer: bucketSizesBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "g1-sparse-bucket-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-sparse-bucket-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  basesBuffer.destroy();
  dummyBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  baseIndicesBuffer.destroy();
  bucketPointersBuffer.destroy();
  bucketSizesBuffer.destroy();

  return unpackPointBatch(result, count);
}

async function runSparseWindowOpBN254(
  device: GPUDevice,
  kernel: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout },
  bucketSums: readonly JacobianPoint[],
  metadata: {
    bucketValues: Uint32Array;
    windowStarts: Uint32Array;
    windowCounts: Uint32Array;
    numWindows: number;
    bucketCount: number;
  },
  count: number,
): Promise<JacobianPoint[]> {
  const totalWindows = count * metadata.numWindows;
  if (totalWindows === 0) {
    return [];
  }
  const aBytes = packPointBatch(bucketSums);
  const outputByteSize = totalWindows * POINT_BYTES;
  const bucketsBuffer = device.createBuffer({ label: "g1-sparse-window-in", size: Math.max(POINT_BYTES, aBytes.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const dummyBuffer = device.createBuffer({ label: "g1-sparse-window-dummy", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ label: "g1-sparse-window-out", size: outputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ label: "g1-sparse-window-staging", size: outputByteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniformBuffer = device.createBuffer({ label: "g1-sparse-window-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bucketValuesBuffer = device.createBuffer({ label: "g1-sparse-window-values", size: Math.max(4, metadata.bucketValues.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const windowStartsBuffer = device.createBuffer({ label: "g1-sparse-window-starts", size: Math.max(4, metadata.windowStarts.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const windowCountsBuffer = device.createBuffer({ label: "g1-sparse-window-counts", size: Math.max(4, metadata.windowCounts.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(bucketsBuffer, 0, aBytes.buffer.slice(aBytes.byteOffset, aBytes.byteOffset + aBytes.byteLength));
  if (metadata.bucketValues.byteLength > 0) {
    device.queue.writeBuffer(bucketValuesBuffer, 0, metadata.bucketValues.buffer.slice(metadata.bucketValues.byteOffset, metadata.bucketValues.byteOffset + metadata.bucketValues.byteLength));
  }
  if (metadata.windowStarts.byteLength > 0) {
    device.queue.writeBuffer(windowStartsBuffer, 0, metadata.windowStarts.buffer.slice(metadata.windowStarts.byteOffset, metadata.windowStarts.byteOffset + metadata.windowStarts.byteLength));
  }
  if (metadata.windowCounts.byteLength > 0) {
    device.queue.writeBuffer(windowCountsBuffer, 0, metadata.windowCounts.buffer.slice(metadata.windowCounts.byteOffset, metadata.windowCounts.byteOffset + metadata.windowCounts.byteLength));
  }
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = totalWindows;
  params[5] = metadata.bucketCount;
  device.queue.writeBuffer(uniformBuffer, 0, params.buffer);

  const bindGroup = device.createBindGroup({
    label: "g1-sparse-window-bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: bucketsBuffer } },
      { binding: 1, resource: { buffer: dummyBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: bucketValuesBuffer } },
      { binding: 5, resource: { buffer: windowStartsBuffer } },
      { binding: 6, resource: { buffer: windowCountsBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "g1-sparse-window-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-sparse-window-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(totalWindows);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  bucketsBuffer.destroy();
  dummyBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  bucketValuesBuffer.destroy();
  windowStartsBuffer.destroy();
  windowCountsBuffer.destroy();

  return unpackPointBatch(result, totalWindows);
}

async function runPippengerMSMSignedSparseBN254(
  device: GPUDevice,
  kernels: {
    bucketSparse: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout };
    windowSparse: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout };
    combine: { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout };
  },
  bases: readonly JacobianPoint[],
  scalars: readonly string[],
  count: number,
  termsPerInstance: number,
  window: number,
): Promise<JacobianPoint[]> {
  const metadata = buildSparseSignedBucketMetadata(scalars, count, termsPerInstance, window);
  const bucketSums = await runSparseBucketOpBN254(device, kernels.bucketSparse, bases, metadata, metadata.bucketPointers.length);
  const windowSums = await runSparseWindowOpBN254(device, kernels.windowSparse, bucketSums, metadata, count);
  return runOpBN254(device, kernels.combine, 0, windowSums, [], {
    count,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
}

async function runSmokeBLS(lines: string[], device: GPUDevice): Promise<void> {
  const [opsShaderText, msmShaderText, vectors] = await Promise.all([
    fetchShaderParts(curveConfig.opsShaderParts),
    fetchShaderParts(curveConfig.msmShaderParts ?? curveConfig.opsShaderParts),
    fetchVectors(),
  ]);
  lines.push("3. Loading shader and vectors... OK");
  lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
  lines.push(`cases.msm = ${vectors.msm_cases.length}`);

  if (!curveConfig.msmShaderParts) {
    throw new Error("missing optimized MSM shader path");
  }

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

  expectPointBatch("msm_naive_affine", await runNaiveMSMBLS(device, kernel, vectors), want);
  lines.push("msm_naive_affine: OK");

  const window = bestPippengerWindow(vectors.terms_per_instance);
  expectPointBatch(
    "msm_pippenger_affine",
    await runOptimizedPippengerMSMBLS(device, optimizedKernels, bases, scalars, vectors.msm_cases.length, vectors.terms_per_instance, window),
    want,
  );
  lines.push(`msm_pippenger_affine (window=${window}): OK`);
}

async function runSmokeBN254(lines: string[], device: GPUDevice): Promise<void> {
  const [shaderText, vectors] = await Promise.all([fetchShaderParts(curveConfig.opsShaderParts), fetchVectors()]);
  lines.push("3. Loading shader and vectors... OK");
  lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
  lines.push(`cases.msm = ${vectors.msm_cases.length}`);

  const kernel = createKernelBN254(device, shaderText);
  const sparseBucketKernel = createKernelBN254(device, shaderText, "g1_msm_bucket_sparse_main");
  const sparseWindowKernel = createKernelBN254(device, shaderText, "g1_msm_window_sparse_main");
  const combineKernel = createKernelBN254(device, shaderText, "g1_msm_combine_main");
  lines.push("4. Creating pipeline... OK");

  const bases: JacobianPoint[] = [];
  const scalars: string[] = [];
  const want: JacobianPoint[] = [];
  for (const item of vectors.msm_cases) {
    for (const base of item.bases_affine) {
      bases.push(affineToKernelPoint(base, vectors.one_mont_z));
    }
    for (const scalar of item.scalars_bytes_le) {
      scalars.push(scalar);
    }
    want.push(item.expected_affine);
  }

  expectPointBatch("msm_naive_affine", await runNaiveMSMBN254(device, kernel, bases, scalars, vectors.msm_cases.length, vectors.terms_per_instance), want);
  lines.push("msm_naive_affine: OK");

  const window = bestPippengerWindow(vectors.terms_per_instance);
  expectPointBatch(
    "msm_pippenger_affine",
    await runPippengerMSMSignedSparseBN254(
      device,
      { bucketSparse: sparseBucketKernel, windowSparse: sparseWindowKernel, combine: combineKernel },
      bases,
      scalars,
      vectors.msm_cases.length,
      vectors.terms_per_instance,
      window,
    ),
    want,
  );
  lines.push(`msm_pippenger_affine (window=${window}): OK`);
}

async function runSmoke(): Promise<void> {
  const lines = [curveConfig.logTitle, ""];
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

    if (curveConfig.id === "bn254") {
      await runSmokeBN254(lines, device);
    } else {
      await runSmokeBLS(lines, device);
    }

    lines.push("");
    lines.push(curveConfig.passLine);
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
