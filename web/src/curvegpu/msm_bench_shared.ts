export type MSMProfile = {
  partitionMs: number;
  uploadMs: number;
  kernelMs: number;
  readbackMs: number;
  scalarMulTotalMs: number;
  bucketReductionTotalMs: number;
  windowReductionTotalMs: number;
  finalReductionTotalMs: number;
  reductionTotalMs: number;
  totalMs: number;
};

export type SizeBenchmarkEntry = {
  label: string;
  window: number;
  run: () => Promise<MSMProfile>;
};

export function zeroMSMProfile(): MSMProfile {
  return {
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
}

export async function benchmarkMSM(
  iters: number,
  run: () => Promise<MSMProfile>,
): Promise<{ cold: MSMProfile; warm: MSMProfile }> {
  const cold = await run();
  if (iters === 1) {
    return { cold, warm: cold };
  }
  const warm = zeroMSMProfile();
  for (let i = 0; i < iters; i += 1) {
    const profile = await run();
    warm.partitionMs += profile.partitionMs;
    warm.uploadMs += profile.uploadMs;
    warm.kernelMs += profile.kernelMs;
    warm.readbackMs += profile.readbackMs;
    warm.scalarMulTotalMs += profile.scalarMulTotalMs;
    warm.bucketReductionTotalMs += profile.bucketReductionTotalMs;
    warm.windowReductionTotalMs += profile.windowReductionTotalMs;
    warm.finalReductionTotalMs += profile.finalReductionTotalMs;
    warm.reductionTotalMs += profile.reductionTotalMs;
    warm.totalMs += profile.totalMs;
  }
  warm.partitionMs /= iters;
  warm.uploadMs /= iters;
  warm.kernelMs /= iters;
  warm.readbackMs /= iters;
  warm.scalarMulTotalMs /= iters;
  warm.bucketReductionTotalMs /= iters;
  warm.windowReductionTotalMs /= iters;
  warm.finalReductionTotalMs /= iters;
  warm.reductionTotalMs /= iters;
  warm.totalMs /= iters;
  return { cold, warm };
}

export function formatMSMBenchmarkRow(options: {
  size: number;
  label: string;
  window: number;
  initMs: number;
  prepMs?: number;
  includePrepMs?: boolean;
  cold: MSMProfile;
  warm: MSMProfile;
}): string {
  const { size, label, window, initMs, prepMs = 0, includePrepMs = false, cold, warm } = options;
  const fields = [
    `${size}`,
    label,
    `${window}`,
    initMs.toFixed(3),
  ];
  if (includePrepMs) {
    fields.push(prepMs.toFixed(3));
  }
  fields.push(
    cold.partitionMs.toFixed(3),
    cold.uploadMs.toFixed(3),
    cold.kernelMs.toFixed(3),
    cold.readbackMs.toFixed(3),
    cold.scalarMulTotalMs.toFixed(3),
    cold.bucketReductionTotalMs.toFixed(3),
    cold.windowReductionTotalMs.toFixed(3),
    cold.finalReductionTotalMs.toFixed(3),
    cold.reductionTotalMs.toFixed(3),
    cold.totalMs.toFixed(3),
    (initMs + prepMs + cold.totalMs).toFixed(3),
    warm.partitionMs.toFixed(3),
    warm.uploadMs.toFixed(3),
    warm.kernelMs.toFixed(3),
    warm.readbackMs.toFixed(3),
    warm.scalarMulTotalMs.toFixed(3),
    warm.bucketReductionTotalMs.toFixed(3),
    warm.windowReductionTotalMs.toFixed(3),
    warm.finalReductionTotalMs.toFixed(3),
    warm.reductionTotalMs.toFixed(3),
    warm.totalMs.toFixed(3),
  );
  return fields.join(",");
}

export async function appendMSMBenchmarkRows(options: {
  lines: string[];
  initMs: number;
  minLog: number;
  maxLog: number;
  iters: number;
  writeLog: (lines: string[]) => void;
  makeSizeBenchmarks: (args: { logSize: number; size: number }) => Promise<{
    prepMs?: number;
    includePrepMs?: boolean;
    entries: SizeBenchmarkEntry[];
  }>;
}): Promise<void> {
  const { lines, initMs, minLog, maxLog, iters, writeLog, makeSizeBenchmarks } = options;
  for (let logSize = minLog; logSize <= maxLog; logSize += 1) {
    const size = 1 << logSize;
    const benchSet = await makeSizeBenchmarks({ logSize, size });
    const prepMs = benchSet.prepMs ?? 0;
    const includePrepMs = benchSet.includePrepMs ?? false;
    for (const bench of benchSet.entries) {
      const benchmark = await benchmarkMSM(iters, bench.run);
      lines.push(formatMSMBenchmarkRow({
        size,
        label: bench.label,
        window: bench.window,
        initMs,
        prepMs,
        includePrepMs,
        cold: benchmark.cold,
        warm: benchmark.warm,
      }));
      writeLog(lines);
    }
  }
}
