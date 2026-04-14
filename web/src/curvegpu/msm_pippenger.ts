import { buildSparseSignedBucketMetadataWords } from "./msm_shared.js";
import {
  createBindGroupForBuffers,
  createEmptyPointStorageBuffer,
  createParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer,
  type Kernel,
  readbackBuffer,
  submitKernel,
} from "./msm_gpu_runtime.js";

type SparseSignedBucketMetadata = ReturnType<typeof buildSparseSignedBucketMetadataWords>;

export type WindowReductionOptions = {
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

export type WindowReductionResult = {
  windowOutput: GPUBuffer;
  cleanupBuffers: GPUBuffer[];
};

export type PippengerRuntime = {
  bucket: Kernel;
  bucketWorkgroupSize?: number;
  combine: Kernel;
  reduceWindows(options: WindowReductionOptions): Promise<WindowReductionResult>;
};

export function buildJacPippengerRuntime(
  kernels: { bucket: Kernel; weightJac: Kernel; subsumJac: Kernel; combine: Kernel },
  workgroupSize = 64,
  debug = false,
): PippengerRuntime {
  return {
    bucket: kernels.bucket,
    bucketWorkgroupSize: workgroupSize,
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
      const weightParams = createParamsBuffer(device, `${labelPrefix}-weight-params`, uniformBytes, { count: bucketCountOut });
      const weightBindGroup = createBindGroupForBuffers(device, kernels.weightJac, `${labelPrefix}-weight-bg`,
        bucketOutput, zeroInput, weightedBucketOutput, weightParams, bucketValuesInput);
      await submitKernel(device, kernels.weightJac, weightBindGroup, bucketCountOut, `${labelPrefix}-weight`, workgroupSize, debug);
      const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
      const windowParams = createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, { count: count * metadata.numWindows });
      const windowBindGroup = createBindGroupForBuffers(device, kernels.subsumJac, `${labelPrefix}-window-bg`,
        weightedBucketOutput, zeroInput, windowOutput, windowParams, bucketValuesInput, windowStartsInput, windowCountsInput);
      await submitKernel(device, kernels.subsumJac, windowBindGroup, count * metadata.numWindows * workgroupSize,
        `${labelPrefix}-window`, workgroupSize, debug);
      return { windowOutput, cleanupBuffers: [weightedBucketOutput, weightParams, windowParams] };
    },
  };
}

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
