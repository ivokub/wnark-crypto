import { buildSparseSignedBucketMetadataWords } from "./msm_shared.js";
import {
  createBindGroupForBuffers,
  createEmptyPointStorageBuffer,
  createMSMKernelSet,
  createParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer,
  Kernel,
  readbackBuffer,
  submitKernel,
} from "./msm_gpu_runtime.js";

type SparseSignedBucketMetadata = ReturnType<typeof buildSparseSignedBucketMetadataWords>;

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
  metadata: SparseSignedBucketMetadata;
  count: number;
  labelPrefix: string;
};

type WindowReductionResult = {
  windowOutput: GPUBuffer;
  cleanupBuffers: GPUBuffer[];
};

export type PippengerRuntime = {
  bucket: Kernel;
  bucketWorkgroupSize?: number;
  combine: Kernel;
  reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult>;
};

export interface PippengerStrategy {
  readonly id: string;
  createRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string, debug?: boolean): PippengerRuntime;
}

function createSimplePippengerRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string, debug = false): PippengerRuntime {
  const kernels = createMSMKernelSet(device, shaderCode, labelPrefix, {
    bucket: "g1_msm_bucket_sparse_main",
    windowSparse: "g1_msm_window_sparse_main",
    combine: "g1_msm_combine_main",
  }, debug);
  return {
    bucket: kernels.bucket,
    combine: kernels.combine,
    async reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult> {
      const {
        device,
        pointBytes,
        uniformBytes,
        zeroInput,
        bucketOutput,
        bucketValuesInput,
        windowStartsInput,
        windowCountsInput,
        metadata,
        count,
        labelPrefix,
      } = options;
      const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
      const windowParams = createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, {
        count: count * metadata.numWindows,
        bucketCount: metadata.bucketCount,
      });
      const windowBindGroup = createBindGroupForBuffers(
        device,
        kernels.windowSparse,
        `${labelPrefix}-window-bg`,
        bucketOutput,
        zeroInput,
        windowOutput,
        windowParams,
        bucketValuesInput,
        windowStartsInput,
        windowCountsInput,
      );
      await submitKernel(
        device,
        kernels.windowSparse,
        windowBindGroup,
        count * metadata.numWindows * 64,
        `${labelPrefix}-window`,
        64,
        debug,
      );
      return {
        windowOutput,
        cleanupBuffers: [windowParams],
      };
    },
  };
}

function createWeightedPippengerRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string, debug = false): PippengerRuntime {
  const kernels = createMSMKernelSet(device, shaderCode, labelPrefix, {
    bucket: "g1_msm_bucket_sparse_main",
    weightBuckets: "g1_msm_weight_buckets_main",
    subsumPhase1: "g1_msm_subsum_phase1_main",
    combine: "g1_msm_combine_main",
  }, debug);
  return {
    bucket: kernels.bucket,
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
      const weightedBucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-weighted-out`, bucketCountOut, pointBytes);
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
      await submitKernel(device, kernels.weightBuckets, weightBindGroup, bucketCountOut, `${labelPrefix}-weight`, 64, debug);

      const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
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
        64,
        debug,
      );
      return {
        windowOutput,
        cleanupBuffers: [weightedBucketOutput, weightParams, windowParams],
      };
    },
  };
}

function createJacPippengerRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string, debug = false): PippengerRuntime {
  const kernels = createMSMKernelSet(device, shaderCode, labelPrefix, {
    bucket: "g1_msm_bucket_jac_main",
    weightJac: "g1_msm_weight_jac_main",
    subsumJac: "g1_msm_subsum_jac_main",
    combine: "g1_msm_combine_jac_main",
  }, debug);
  return {
    bucket: kernels.bucket,
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

      const weightedBucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-jac-weighted-out`, bucketCountOut, pointBytes);
      const weightParams = createParamsBuffer(device, `${labelPrefix}-jac-weight-params`, uniformBytes, {
        count: bucketCountOut,
      });
      const weightBindGroup = createBindGroupForBuffers(
        device,
        kernels.weightJac,
        `${labelPrefix}-jac-weight-bg`,
        bucketOutput,
        zeroInput,
        weightedBucketOutput,
        weightParams,
        bucketValuesInput,
      );
      await submitKernel(device, kernels.weightJac, weightBindGroup, bucketCountOut, `${labelPrefix}-jac-weight`, 64, debug);

      const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-jac-window-out`, count * metadata.numWindows, pointBytes);
      const windowParams = createParamsBuffer(device, `${labelPrefix}-jac-window-params`, uniformBytes, {
        count: count * metadata.numWindows,
      });
      const windowBindGroup = createBindGroupForBuffers(
        device,
        kernels.subsumJac,
        `${labelPrefix}-jac-window-bg`,
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
        kernels.subsumJac,
        windowBindGroup,
        count * metadata.numWindows * 64,
        `${labelPrefix}-jac-window`,
        64,
        debug,
      );

      return {
        windowOutput,
        cleanupBuffers: [weightedBucketOutput, weightParams, windowParams],
      };
    },
  };
}

export const simpleSparsePippengerStrategy: PippengerStrategy = {
  id: "simple-sparse",
  createRuntime: createSimplePippengerRuntime,
};

export const weightedSparsePippengerStrategy: PippengerStrategy = {
  id: "weighted-sparse",
  createRuntime: createWeightedPippengerRuntime,
};

export const jacSparsePippengerStrategy: PippengerStrategy = {
  id: "jac-sparse",
  createRuntime: createJacPippengerRuntime,
};

export async function runSparseSignedPippengerMSM(options: {
  device: GPUDevice;
  runtime: PippengerRuntime;
  basesBytes: Uint8Array;
  pointBytes: number;
  uniformBytes: number;
  zeroPointBytes: Uint8Array;
  scalarWords: Uint32Array;
  count: number;
  termsPerInstance: number;
  window: number;
  maxChunkSize?: number;
  labelPrefix: string;
  debug?: boolean;
}): Promise<Uint8Array> {
  const {
    device,
    runtime,
    basesBytes,
    pointBytes,
    uniformBytes,
    zeroPointBytes,
    scalarWords,
    count,
    termsPerInstance,
    window,
    maxChunkSize = 256,
    labelPrefix,
    debug = false,
  } = options;

  const metadata = buildSparseSignedBucketMetadataWords(scalarWords, count, termsPerInstance, window, maxChunkSize);
  if (debug) {
    console.debug("[curvegpu] msm metadata", {
    labelPrefix,
    count,
    termsPerInstance,
    window,
    pointBytes,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
    bucketCountOut: metadata.bucketPointers.length,
    baseIndicesLen: metadata.baseIndices.length,
    bucketPointersLen: metadata.bucketPointers.length,
    bucketSizesLen: metadata.bucketSizes.length,
    bucketValuesLen: metadata.bucketValues.length,
    windowStartsLen: metadata.windowStarts.length,
    windowCountsLen: metadata.windowCounts.length,
    bucketSizesHead: Array.from(metadata.bucketSizes.slice(0, 16)),
    bucketValuesHead: Array.from(metadata.bucketValues.slice(0, 16)),
    windowCountsHead: Array.from(metadata.windowCounts.slice(0, 16)),
    });
  }
  const zeroInput = createStorageBufferFromBytes(device, `${labelPrefix}-zero`, zeroPointBytes, pointBytes);
  const basesInput = createStorageBufferFromBytes(
    device,
    `${labelPrefix}-bases`,
    basesBytes,
    Math.max(1, termsPerInstance * count) * pointBytes,
  );
  const baseIndicesInput = createU32StorageBuffer(device, `${labelPrefix}-base-indices`, metadata.baseIndices);
  const bucketPointersInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-pointers`, metadata.bucketPointers);
  const bucketSizesInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-sizes`, metadata.bucketSizes);

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-bucket-out`, bucketCountOut, pointBytes);
  const bucketParams = createParamsBuffer(device, `${labelPrefix}-bucket-params`, uniformBytes, {
    count: bucketCountOut,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  const bucketBindGroup = createBindGroupForBuffers(
    device,
    runtime.bucket,
    `${labelPrefix}-bucket-bg`,
    basesInput,
    zeroInput,
    bucketOutput,
    bucketParams,
    baseIndicesInput,
    bucketPointersInput,
    bucketSizesInput,
  );
  await submitKernel(device, runtime.bucket, bucketBindGroup, bucketCountOut, `${labelPrefix}-bucket`, runtime.bucketWorkgroupSize ?? 64, debug);

  const bucketValuesInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-values`, metadata.bucketValues);
  const windowStartsInput = createU32StorageBuffer(device, `${labelPrefix}-window-starts`, metadata.windowStarts);
  const windowCountsInput = createU32StorageBuffer(device, `${labelPrefix}-window-counts`, metadata.windowCounts);

  const { windowOutput, cleanupBuffers: windowReductionCleanup } = await runtime.reduceWindows({
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
  });

  const finalOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-final-out`, count, pointBytes);
  const finalParams = createParamsBuffer(device, `${labelPrefix}-final-params`, uniformBytes, {
    count,
    termsPerInstance,
    window,
    numWindows: metadata.numWindows,
    bucketCount: metadata.bucketCount,
  });
  const finalBindGroup = createBindGroupForBuffers(
    device,
    runtime.combine,
    `${labelPrefix}-final-bg`,
    windowOutput,
    zeroInput,
    finalOutput,
    finalParams,
  );
  await submitKernel(device, runtime.combine, finalBindGroup, count, `${labelPrefix}-final`, 64, debug);

  const result = await readbackBuffer(device, finalOutput, Math.max(1, count) * pointBytes);

  zeroInput.destroy();
  basesInput.destroy();
  baseIndicesInput.destroy();
  bucketPointersInput.destroy();
  bucketSizesInput.destroy();
  bucketValuesInput.destroy();
  windowStartsInput.destroy();
  windowCountsInput.destroy();
  bucketOutput.destroy();
  windowReductionCleanup.forEach((buffer) => buffer.destroy());
  windowOutput.destroy();
  finalOutput.destroy();
  bucketParams.destroy();
  finalParams.destroy();

  return result;
}
