import type {
  CurveGPUContext,
  CurveGPUMSMOptions,
  CurveGPUPackedPointLayout,
  SupportedCurveID,
} from "./api.js";
import { createBindGroupForBuffers, createEmptyPointStorageBuffer, createMSMKernelSetAsync, createParamsBuffer, createStorageBufferFromBytes, createU32StorageBuffer, submitKernel, type Kernel } from "./msm_gpu_runtime.js";
import { runSparseSignedPippengerMSM } from "./msm_pippenger.js";
import { bestPippengerWindow } from "./msm_shared.js";
import { lazyAsync, loadShaderParts } from "./runtime_common.js";

type WindowReductionOptions = {
  device: GPUDevice;
  pointBytes: number;
  uniformBytes: number;
  zeroInput: GPUBuffer;
  bucketOutput: GPUBuffer;
  bucketCountOut: number;
  bucketValuesInput: GPUBuffer;
  windowStartsInput: GPUBuffer;
  windowCountsInput: GPUBuffer;
  metadata: {
    numWindows: number;
    bucketCount: number;
  };
  count: number;
  labelPrefix: string;
};

type WindowReductionResult = {
  windowOutput: GPUBuffer;
  cleanupBuffers: GPUBuffer[];
};

type PippengerRuntime = {
  bucket: Kernel;
  bucketWorkgroupSize?: number;
  combine: Kernel;
  reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult>;
};

export interface OptimizedG2MSMBenchModule {
  bestWindow(termCount: number): number;
  pippengerPackedJacobianBases(
    basesPacked: Uint8Array,
    scalarsPacked: Uint8Array,
    options: CurveGPUMSMOptions & { layout?: CurveGPUPackedPointLayout },
  ): Promise<Uint8Array>;
}

function ensurePackedScalars(scalarsPacked: Uint8Array, count: number, label: string): void {
  const expected = count * 32;
  if (scalarsPacked.byteLength !== expected) {
    throw new Error(`${label}: expected ${expected} scalar bytes, got ${scalarsPacked.byteLength}`);
  }
}

function packScalarWordsPacked(scalarsPacked: Uint8Array): Uint32Array {
  if (scalarsPacked.byteLength % 32 !== 0) {
    throw new Error(`packed scalars: expected a multiple of 32 bytes, got ${scalarsPacked.byteLength}`);
  }
  const out = new Uint32Array((scalarsPacked.byteLength / 32) * 8);
  const view = new DataView(scalarsPacked.buffer, scalarsPacked.byteOffset, scalarsPacked.byteLength);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = view.getUint32(i * 4, true);
  }
  return out;
}

function shaderPartsForCurve(curve: SupportedCurveID): readonly string[] {
  if (curve === "bn254") {
    return [
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-types",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-consts",
      "/shaders/curves/bn254/fp_arith.wgsl#section=fp-core",
      "/shaders/curves/bn254/g2_arith.wgsl",
      "/shaders/common/g2_msm.wgsl",
    ];
  }
  return [
    "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-types",
    "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-consts",
    "/shaders/curves/bls12_381/fp_arith.wgsl#section=fp-core",
    "/shaders/curves/bls12_381/g2_arith.wgsl",
    "/shaders/common/g2_msm.wgsl",
  ];
}

async function createWeightedG2PippengerRuntime(
  device: GPUDevice,
  shaderCode: string,
  labelPrefix: string,
): Promise<PippengerRuntime> {
  const kernels = await createMSMKernelSetAsync(device, shaderCode, labelPrefix, {
    bucket: "g2_msm_bucket_sparse_main",
    weightBuckets: "g2_msm_weight_buckets_main",
    subsumPhase1: "g2_msm_subsum_phase1_main",
    combine: "g2_msm_combine_main",
  });
  return {
    bucket: kernels.bucket,
    bucketWorkgroupSize: 32,
    combine: kernels.combine,
    async reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult> {
      const {
        device,
        pointBytes,
        uniformBytes,
        zeroInput,
        bucketOutput,
        bucketCountOut,
        bucketValuesInput,
        windowStartsInput,
        windowCountsInput,
        metadata,
        count,
        labelPrefix,
      } = options;
      const weightedBucketOutput = createEmptyPointStorageBuffer(
        device,
        `${labelPrefix}-weighted-out`,
        bucketCountOut,
        pointBytes,
      );
      const weightParams = createParamsBuffer(device, `${labelPrefix}-weight-params`, uniformBytes, {
        count: bucketCountOut,
      });
      const weightBindGroup = createBindGroupForBuffers(
        device,
        kernels.weightBuckets,
        `${labelPrefix}-weight-bg`,
        bucketOutput,
        zeroInput,
        weightedBucketOutput,
        weightParams,
        bucketValuesInput,
      );
      await submitKernel(device, kernels.weightBuckets, weightBindGroup, bucketCountOut, `${labelPrefix}-weight`);

      const windowOutput = createEmptyPointStorageBuffer(
        device,
        `${labelPrefix}-window-out`,
        count * metadata.numWindows,
        pointBytes,
      );
      const windowParams = createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, {
        count: count * metadata.numWindows,
      });
      const windowBindGroup = createBindGroupForBuffers(
        device,
        kernels.subsumPhase1,
        `${labelPrefix}-window-bg`,
        weightedBucketOutput,
        zeroInput,
        windowOutput,
        windowParams,
        bucketValuesInput,
        windowStartsInput,
        windowCountsInput,
      );
      await submitKernel(
        device,
        kernels.subsumPhase1,
        windowBindGroup,
        count * metadata.numWindows * 64,
        `${labelPrefix}-window`,
      );
      return {
        windowOutput,
        cleanupBuffers: [weightedBucketOutput, weightParams, windowParams],
      };
    },
  };
}

export function createOptimizedG2MSMBenchModule(
  context: CurveGPUContext,
  curve: SupportedCurveID,
  pointBytes: number,
): OptimizedG2MSMBenchModule {
  const label = `${curve}-g2-msm-bench`;
  const getRuntime = lazyAsync(async (): Promise<PippengerRuntime> => {
    const shaderCode = await loadShaderParts(shaderPartsForCurve(curve));
    return createWeightedG2PippengerRuntime(context.device, shaderCode, label);
  });

  return {
    bestWindow(termCount: number): number {
      return bestPippengerWindow(termCount);
    },
    async pippengerPackedJacobianBases(
      basesPacked: Uint8Array,
      scalarsPacked: Uint8Array,
      options: CurveGPUMSMOptions & { layout?: CurveGPUPackedPointLayout },
    ): Promise<Uint8Array> {
      const count = options.count ?? 1;
      const termsPerInstance = options.termsPerInstance ?? 0;
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`${label}: count must be a positive integer`);
      }
      if (!Number.isInteger(termsPerInstance) || termsPerInstance <= 0) {
        throw new Error(`${label}: termsPerInstance must be a positive integer`);
      }
      if ((options.layout ?? "jacobian_x_y_z_le") !== "jacobian_x_y_z_le") {
        throw new Error(`${label}: unsupported packed point layout ${options.layout}`);
      }
      const expectedPointBytes = count * termsPerInstance * pointBytes;
      if (basesPacked.byteLength !== expectedPointBytes) {
        throw new Error(`${label}: expected ${expectedPointBytes} base bytes, got ${basesPacked.byteLength}`);
      }
      ensurePackedScalars(scalarsPacked, count * termsPerInstance, `${label}.scalarsPacked`);

      const runtime = await getRuntime();
      const window = options.window ?? bestPippengerWindow(termsPerInstance);
      return runSparseSignedPippengerMSM({
        device: context.device,
        runtime,
        basesBytes: basesPacked,
        pointBytes,
        uniformBytes: 32,
        zeroPointBytes: new Uint8Array(pointBytes),
        scalarWords: packScalarWordsPacked(scalarsPacked),
        count,
        termsPerInstance,
        window,
        maxChunkSize: options.maxChunkSize,
        labelPrefix: label,
      });
    },
  };
}
