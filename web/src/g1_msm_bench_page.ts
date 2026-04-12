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
  makeRandomScalarBatch,
} from "./curvegpu/msm_shared.js";
import {
  appendMSMBenchmarkRows,
  benchmarkMSM,
  MSMProfile,
} from "./curvegpu/msm_bench_shared.js";
import { runMSMBenchmarkPage } from "./curvegpu/msm_page_runner.js";
import { runSparseSignedPippengerMSMBenchmark, type PippengerBenchKernels } from "./curvegpu/msm_pippenger_bench.js";
import { createPreferredByteBaseSource } from "./curvegpu/msm_bench_sources.js";
import {
  createMSMKernel as sharedCreateMSMKernel,
  createMSMKernelSet,
  Kernel,
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

type OpBenchmark = {
  uploadMs: number;
  kernelMs: number;
  readbackMs: number;
  totalMs: number;
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
const G1_OP_JAC_INFINITY = 1;
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;

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

function bestPippengerWindow(count: number): number {
  return sharedBestPippengerWindow(count);
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

async function runOpBenchmark(
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
): Promise<{ out: JacobianPoint[]; profile: OpBenchmark }> {
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

async function runScalarMulAffineBenchmark(
  device: GPUDevice,
  kernel: Kernel,
  base: readonly JacobianPoint[],
  scalarsHexLE: readonly string[],
): Promise<{ out: JacobianPoint[]; profile: MSMProfile }> {
  const totalStart = performance.now();
  let uploadMs = 0;
  let kernelMs = 0;
  let readbackMs = 0;
  let acc = await runOpBenchmark(device, kernel, G1_OP_JAC_INFINITY, base.map(() => zeroPoint()), base.map(() => zeroPoint()));
  uploadMs += acc.profile.uploadMs;
  kernelMs += acc.profile.kernelMs;
  readbackMs += acc.profile.readbackMs;
  for (let bit = 255; bit >= 0; bit -= 1) {
    const doubled = await runOpBenchmark(device, kernel, G1_OP_DOUBLE_JAC, acc.out, acc.out.map(() => zeroPoint()));
    uploadMs += doubled.profile.uploadMs;
    kernelMs += doubled.profile.kernelMs;
    readbackMs += doubled.profile.readbackMs;
    acc = doubled;
    const mask = scalarsHexLE.map((scalar) => scalarBit(scalar, bit));
    if (!anyMask(mask)) {
      continue;
    }
    const added = await runOpBenchmark(device, kernel, G1_OP_ADD_MIXED, acc.out, maskedAffine(base, mask));
    uploadMs += added.profile.uploadMs;
    kernelMs += added.profile.kernelMs;
    readbackMs += added.profile.readbackMs;
    acc = added;
  }
  const affine = await runOpBenchmark(device, kernel, G1_OP_JAC_TO_AFFINE, acc.out, acc.out.map(() => zeroPoint()));
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
  const generated = await runScalarMulAffineBenchmark(device, kernel, bases, scalars);
  return packPointBatch(generated.out);
}

async function runPippengerMSMBenchmark(
  device: GPUDevice,
  kernels: PippengerBenchKernels,
  basesBytes: Uint8Array,
  scalars: ScalarBatch,
  window: number,
): Promise<MSMProfile> {
  return runSparseSignedPippengerMSMBenchmark({
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
                  runPippengerMSMBenchmark(
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
