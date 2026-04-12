import { buildSparseSignedBucketMetadataWords } from "./msm_shared.js";
import {
  createBindGroupForBuffers,
  createEmptyPointStorageBuffer,
  createParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer,
  Kernel,
  readbackBuffer,
  submitKernel,
} from "./msm_gpu_runtime.js";

type PippengerSimpleKernels = {
  bucket: Kernel;
  windowSparse: Kernel;
  combine: Kernel;
};

type PippengerWeightedKernels = {
  bucket: Kernel;
  weightBuckets: Kernel;
  subsumPhase1: Kernel;
  combine: Kernel;
};

export type PippengerKernels =
  | ({ mode: "simple" } & PippengerSimpleKernels)
  | ({ mode: "weighted" } & PippengerWeightedKernels);

export async function runSparseSignedPippengerMSM(options: {
  device: GPUDevice;
  kernels: PippengerKernels;
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
    kernels,
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
    kernels.bucket,
    `${labelPrefix}-bucket-bg`,
    basesInput,
    zeroInput,
    bucketOutput,
    bucketParams,
    baseIndicesInput,
    bucketPointersInput,
    bucketSizesInput,
  );
  await submitKernel(device, kernels.bucket, bucketBindGroup, bucketCountOut, `${labelPrefix}-bucket`);

  const bucketValuesInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-values`, metadata.bucketValues);
  const windowStartsInput = createU32StorageBuffer(device, `${labelPrefix}-window-starts`, metadata.windowStarts);
  const windowCountsInput = createU32StorageBuffer(device, `${labelPrefix}-window-counts`, metadata.windowCounts);

  let windowInput = bucketOutput;
  let weightedBucketOutput: GPUBuffer | null = null;
  let weightParams: GPUBuffer | null = null;
  if (kernels.mode === "weighted") {
    weightedBucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-weighted-out`, bucketCountOut, pointBytes);
    weightParams = createParamsBuffer(device, `${labelPrefix}-weight-params`, uniformBytes, {
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
    windowInput = weightedBucketOutput;
  }

  const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
  const windowParams = createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, {
    count: count * metadata.numWindows,
    bucketCount: kernels.mode === "simple" ? metadata.bucketCount : 0,
  });
  const windowKernel = kernels.mode === "simple" ? kernels.windowSparse : kernels.subsumPhase1;
  const windowBindGroup = createBindGroupForBuffers(
    device,
    windowKernel,
    `${labelPrefix}-window-bg`,
    windowInput,
    zeroInput,
    windowOutput,
    windowParams,
    bucketValuesInput,
    windowStartsInput,
    windowCountsInput,
  );
  await submitKernel(
    device,
    windowKernel,
    windowBindGroup,
    count * metadata.numWindows * 64,
    `${labelPrefix}-window`,
  );

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
    kernels.combine,
    `${labelPrefix}-final-bg`,
    windowOutput,
    zeroInput,
    finalOutput,
    finalParams,
  );
  await submitKernel(device, kernels.combine, finalBindGroup, count, `${labelPrefix}-final`);

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
  weightedBucketOutput?.destroy();
  windowOutput.destroy();
  finalOutput.destroy();
  bucketParams.destroy();
  weightParams?.destroy();
  windowParams.destroy();
  finalParams.destroy();

  return result;
}
