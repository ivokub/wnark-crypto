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
  combine: Kernel;
  reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult>;
};

export interface PippengerStrategy {
  readonly id: string;
  createRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string): PippengerRuntime;
}

function createSimplePippengerRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string): PippengerRuntime {
  const kernels = createMSMKernelSet(device, shaderCode, labelPrefix, {
    bucket: "g1_msm_bucket_sparse_main",
    windowSparse: "g1_msm_window_sparse_main",
    combine: "g1_msm_combine_main",
  });
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
      );
      return {
        windowOutput,
        cleanupBuffers: [windowParams],
      };
    },
  };
}

function createWeightedPippengerRuntime(device: GPUDevice, shaderCode: string, labelPrefix: string): PippengerRuntime {
  const kernels = createMSMKernelSet(device, shaderCode, labelPrefix, {
    bucket: "g1_msm_bucket_sparse_main",
    weightBuckets: "g1_msm_weight_buckets_main",
    subsumPhase1: "g1_msm_subsum_phase1_main",
    combine: "g1_msm_combine_main",
  });
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
      await submitKernel(device, kernels.weightBuckets, weightBindGroup, bucketCountOut, `${labelPrefix}-weight`);

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
  } = options;

  const metadata = buildSparseSignedBucketMetadataWords(scalarWords, count, termsPerInstance, window, maxChunkSize);
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
  await submitKernel(device, runtime.bucket, bucketBindGroup, bucketCountOut, `${labelPrefix}-bucket`);

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
  await submitKernel(device, runtime.combine, finalBindGroup, count, `${labelPrefix}-final`);

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
