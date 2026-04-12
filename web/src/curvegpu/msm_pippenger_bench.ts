import type { MSMProfile } from "./msm_bench_shared.js";
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

export type PippengerBenchKernels =
  | ({ mode: "simple" } & PippengerSimpleKernels)
  | ({ mode: "weighted" } & PippengerWeightedKernels);

function measureSync<T>(fn: () => T): { value: T; ms: number } {
  const start = performance.now();
  return { value: fn(), ms: performance.now() - start };
}

async function measureAsync<T>(fn: () => Promise<T>): Promise<{ value: T; ms: number }> {
  const start = performance.now();
  return { value: await fn(), ms: performance.now() - start };
}

export async function runSparseSignedPippengerMSMBenchmark(options: {
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

  const zeroInput = measureSync(() => createStorageBufferFromBytes(device, `${labelPrefix}-zero`, zeroPointBytes, pointBytes));
  const basesInput = measureSync(() =>
    createStorageBufferFromBytes(device, `${labelPrefix}-bases`, basesBytes, Math.max(1, termsPerInstance) * pointBytes),
  );
  const baseIndicesInput = measureSync(() => createU32StorageBuffer(device, `${labelPrefix}-base-indices`, metadata.baseIndices));
  const bucketPointersInput = measureSync(() =>
    createU32StorageBuffer(device, `${labelPrefix}-bucket-pointers`, metadata.bucketPointers),
  );
  const bucketSizesInput = measureSync(() => createU32StorageBuffer(device, `${labelPrefix}-bucket-sizes`, metadata.bucketSizes));
  profile.uploadMs += zeroInput.ms + basesInput.ms + baseIndicesInput.ms + bucketPointersInput.ms + bucketSizesInput.ms;

  const bucketCountOut = metadata.bucketPointers.length;
  const bucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-bucket-out`, bucketCountOut, pointBytes);
  const bucketParams = measureSync(() =>
    createParamsBuffer(device, `${labelPrefix}-bucket-params`, uniformBytes, {
      count: bucketCountOut,
      termsPerInstance,
      window,
      numWindows: metadata.numWindows,
      bucketCount: metadata.bucketCount,
    }),
  );
  profile.uploadMs += bucketParams.ms;
  const bucketBindGroup = createBindGroupForBuffers(
    device,
    kernels.bucket,
    `${labelPrefix}-bucket-bg`,
    basesInput.value,
    zeroInput.value,
    bucketOutput,
    bucketParams.value,
    baseIndicesInput.value,
    bucketPointersInput.value,
    bucketSizesInput.value,
  );
  const bucketStart = performance.now();
  const bucketKernel = await measureAsync(() =>
    submitKernel(device, kernels.bucket, bucketBindGroup, bucketCountOut, `${labelPrefix}-bucket`),
  );
  profile.kernelMs += bucketKernel.ms;
  profile.bucketReductionTotalMs += performance.now() - bucketStart;

  const bucketValuesInput = measureSync(() => createU32StorageBuffer(device, `${labelPrefix}-bucket-values`, metadata.bucketValues));
  const windowStartsInput = measureSync(() =>
    createU32StorageBuffer(device, `${labelPrefix}-window-starts`, metadata.windowStarts),
  );
  const windowCountsInput = measureSync(() => createU32StorageBuffer(device, `${labelPrefix}-window-counts`, metadata.windowCounts));
  profile.uploadMs += bucketValuesInput.ms + windowStartsInput.ms + windowCountsInput.ms;

  let windowInput = bucketOutput;
  if (kernels.mode === "weighted") {
    const weightedBucketOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-weighted-out`, bucketCountOut, pointBytes);
    const weightParams = measureSync(() =>
      createParamsBuffer(device, `${labelPrefix}-weight-params`, uniformBytes, {
        count: bucketCountOut,
      }),
    );
    profile.uploadMs += weightParams.ms;
    const weightBindGroup = createBindGroupForBuffers(
      device,
      kernels.weightBuckets,
      `${labelPrefix}-weight-bg`,
      bucketOutput,
      zeroInput.value,
      weightedBucketOutput,
      weightParams.value,
      bucketValuesInput.value,
    );
    const weightStart = performance.now();
    const weightKernel = await measureAsync(() =>
      submitKernel(device, kernels.weightBuckets, weightBindGroup, bucketCountOut, `${labelPrefix}-weight`),
    );
    profile.kernelMs += weightKernel.ms;
    profile.windowReductionTotalMs += performance.now() - weightStart;
    bucketOutput.destroy();
    weightParams.value.destroy();
    windowInput = weightedBucketOutput;
  }

  const windowOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-window-out`, count * metadata.numWindows, pointBytes);
  const windowParams = measureSync(() =>
    createParamsBuffer(device, `${labelPrefix}-window-params`, uniformBytes, {
      count: count * metadata.numWindows,
      bucketCount: kernels.mode === "simple" ? metadata.bucketCount : 0,
    }),
  );
  profile.uploadMs += windowParams.ms;
  const windowKernel = kernels.mode === "simple" ? kernels.windowSparse : kernels.subsumPhase1;
  const windowBindGroup = createBindGroupForBuffers(
    device,
    windowKernel,
    `${labelPrefix}-window-bg`,
    windowInput,
    zeroInput.value,
    windowOutput,
    windowParams.value,
    bucketValuesInput.value,
    windowStartsInput.value,
    windowCountsInput.value,
  );
  const windowStart = performance.now();
  const windowKernelRun = await measureAsync(() =>
    submitKernel(device, windowKernel, windowBindGroup, count * metadata.numWindows * 64, `${labelPrefix}-window`),
  );
  profile.kernelMs += windowKernelRun.ms;
  profile.windowReductionTotalMs += performance.now() - windowStart;

  const finalOutput = createEmptyPointStorageBuffer(device, `${labelPrefix}-final-out`, count, pointBytes);
  const finalParams = measureSync(() =>
    createParamsBuffer(device, `${labelPrefix}-final-params`, uniformBytes, {
      count,
      termsPerInstance,
      window,
      numWindows: metadata.numWindows,
      bucketCount: metadata.bucketCount,
    }),
  );
  profile.uploadMs += finalParams.ms;
  const finalBindGroup = createBindGroupForBuffers(
    device,
    kernels.combine,
    `${labelPrefix}-final-bg`,
    windowOutput,
    zeroInput.value,
    finalOutput,
    finalParams.value,
  );
  const finalStart = performance.now();
  const finalKernel = await measureAsync(() =>
    submitKernel(device, kernels.combine, finalBindGroup, count, `${labelPrefix}-final`),
  );
  profile.kernelMs += finalKernel.ms;
  profile.finalReductionTotalMs += performance.now() - finalStart;

  const readback = await measureAsync(() => readbackBuffer(device, finalOutput, Math.max(1, count) * pointBytes));
  profile.readbackMs += readback.ms;

  basesInput.value.destroy();
  zeroInput.value.destroy();
  baseIndicesInput.value.destroy();
  bucketPointersInput.value.destroy();
  bucketSizesInput.value.destroy();
  bucketValuesInput.value.destroy();
  windowStartsInput.value.destroy();
  windowCountsInput.value.destroy();
  windowInput.destroy();
  bucketParams.value.destroy();
  windowOutput.destroy();
  windowParams.value.destroy();
  finalOutput.destroy();
  finalParams.value.destroy();

  profile.reductionTotalMs =
    profile.bucketReductionTotalMs + profile.windowReductionTotalMs + profile.finalReductionTotalMs;
  profile.totalMs = performance.now() - totalStart;
  return profile;
}
