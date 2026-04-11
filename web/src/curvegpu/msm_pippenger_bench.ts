import type { MSMProfile } from "./msm_bench_shared.js";
import { buildSparseSignedBucketMetadataWords } from "./msm_shared.js";
import {
  createBindGroupForBuffers,
  createEmptyPointStorageBuffer,
  createParamsBuffer,
  createStorageBufferFromBytes,
  createU32StorageBuffer,
  Kernel,
  readbackBufferProfiled,
  submitKernelProfiled,
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

export type PippengerBenchKernels =
  | ({ mode: "simple" } & PippengerSimpleKernels)
  | ({ mode: "weighted" } & PippengerWeightedKernels);

export async function runSparseSignedPippengerMSMProfiled(options: {
  device: GPUDevice;
  kernels: PippengerBenchKernels;
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
}): Promise<MSMProfile> {
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
  const totalStart = performance.now();
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
  const metadata = buildSparseSignedBucketMetadataWords(scalarWords, count, termsPerInstance, window, maxChunkSize);
  profile.partitionMs = performance.now() - partitionStart;

  const zeroInput = createStorageBufferFromBytes(device, `${labelPrefix}-zero`, zeroPointBytes, pointBytes);
  const basesInput = createStorageBufferFromBytes(
    device,
    `${labelPrefix}-bases`,
    basesBytes,
    Math.max(1, termsPerInstance) * pointBytes,
  );
  const baseIndicesInput = createU32StorageBuffer(device, `${labelPrefix}-base-indices`, metadata.baseIndices);
  const bucketPointersInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-pointers`, metadata.bucketPointers);
  const bucketSizesInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-sizes`, metadata.bucketSizes);
  profile.uploadMs +=
    zeroInput.uploadMs +
    basesInput.uploadMs +
    baseIndicesInput.uploadMs +
    bucketPointersInput.uploadMs +
    bucketSizesInput.uploadMs;

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-bucket-out`, bucketCountOut, pointBytes);
  const bucketParams = createParamsBuffer(device, `${labelPrefix}-bucket-params`, uniformBytes, {
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
    `${labelPrefix}-bucket-bg`,
    basesInput.buffer,
    zeroInput.buffer,
    bucketOutput,
    bucketParams.buffer,
    baseIndicesInput.buffer,
    bucketPointersInput.buffer,
    bucketSizesInput.buffer,
  );
  const bucketStart = performance.now();
  const bucketKernelMs = await submitKernelProfiled(device, kernels.bucket, bucketBindGroup, bucketCountOut, `${labelPrefix}-bucket`);
  profile.kernelMs += bucketKernelMs;
  profile.bucketReductionTotalMs += performance.now() - bucketStart;

  const bucketValuesInput = createU32StorageBuffer(device, `${labelPrefix}-bucket-values`, metadata.bucketValues);
  const windowStartsInput = createU32StorageBuffer(device, `${labelPrefix}-window-starts`, metadata.windowStarts);
  const windowCountsInput = createU32StorageBuffer(device, `${labelPrefix}-window-counts`, metadata.windowCounts);
  profile.uploadMs += bucketValuesInput.uploadMs + windowStartsInput.uploadMs + windowCountsInput.uploadMs;

  let windowInput = bucketOutput;
  if (kernels.mode === "weighted") {
    const weightedBucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-weighted-out`, bucketCountOut, pointBytes);
    const weightParams = createParamsBuffer(device, `${labelPrefix}-weight-params`, uniformBytes, {
      count: bucketCountOut,
    });
    profile.uploadMs += weightParams.uploadMs;
    const weightBindGroup = createBindGroupForBuffers(
      device,
      kernels.weightBuckets,
      `${labelPrefix}-weight-bg`,
      bucketOutput,
      zeroInput.buffer,
      weightedBucketOutput,
      weightParams.buffer,
      bucketValuesInput.buffer,
    );
    const weightStart = performance.now();
    const weightKernelMs = await submitKernelProfiled(
      device,
      kernels.weightBuckets,
      weightBindGroup,
      bucketCountOut,
      `${labelPrefix}-weight`,
    );
    profile.kernelMs += weightKernelMs;
    profile.windowReductionTotalMs += performance.now() - weightStart;
    bucketOutput.destroy();
    weightParams.buffer.destroy();
    windowInput = weightedBucketOutput;
  }

  const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
  const windowParams = createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, {
    count: count * metadata.numWindows,
    bucketCount: kernels.mode === "simple" ? metadata.bucketCount : 0,
  });
  profile.uploadMs += windowParams.uploadMs;
  const windowKernel = kernels.mode === "simple" ? kernels.windowSparse : kernels.subsumPhase1;
  const windowBindGroup = createBindGroupForBuffers(
    device,
    windowKernel,
    `${labelPrefix}-window-bg`,
    windowInput,
    zeroInput.buffer,
    windowOutput,
    windowParams.buffer,
    bucketValuesInput.buffer,
    windowStartsInput.buffer,
    windowCountsInput.buffer,
  );
  const windowStart = performance.now();
  const windowKernelMs = await submitKernelProfiled(
    device,
    windowKernel,
    windowBindGroup,
    count * metadata.numWindows * 64,
    `${labelPrefix}-window`,
  );
  profile.kernelMs += windowKernelMs;
  profile.windowReductionTotalMs += performance.now() - windowStart;

  const finalOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-final-out`, count, pointBytes);
  const finalParams = createParamsBuffer(device, `${labelPrefix}-final-params`, uniformBytes, {
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
    `${labelPrefix}-final-bg`,
    windowOutput,
    zeroInput.buffer,
    finalOutput,
    finalParams.buffer,
  );
  const finalStart = performance.now();
  const finalKernelMs = await submitKernelProfiled(device, kernels.combine, finalBindGroup, count, `${labelPrefix}-final`);
  profile.kernelMs += finalKernelMs;
  profile.finalReductionTotalMs += performance.now() - finalStart;

  const readback = await readbackBufferProfiled(device, finalOutput, Math.max(1, count) * pointBytes);
  profile.readbackMs += readback.readbackMs;

  basesInput.buffer.destroy();
  zeroInput.buffer.destroy();
  baseIndicesInput.buffer.destroy();
  bucketPointersInput.buffer.destroy();
  bucketSizesInput.buffer.destroy();
  bucketValuesInput.buffer.destroy();
  windowStartsInput.buffer.destroy();
  windowCountsInput.buffer.destroy();
  windowInput.destroy();
  bucketParams.buffer.destroy();
  windowOutput.destroy();
  windowParams.buffer.destroy();
  finalOutput.destroy();
  finalParams.buffer.destroy();

  profile.reductionTotalMs =
    profile.bucketReductionTotalMs + profile.windowReductionTotalMs + profile.finalReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return profile;
}
