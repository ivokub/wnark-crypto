import type {
  CurveGPUAffinePoint,
  CurveGPUContext,
  CurveGPUElementBytes,
  CurveGPUJacobianPoint,
  CurveGPUPackedPointLayout,
  FieldModule,
  MSMModule,
  SupportedCurveID,
  CurveGPUMSMOptions,
} from "./api.js";
import { splitBytesLEToU32 } from "./convert.js";
import { bestPippengerWindow } from "./msm_shared.js";
import { runSparseSignedPippengerMSM } from "./msm_pippenger.js";
import type { PippengerRuntime, PippengerStrategy } from "./msm_pippenger.js";
import { cloneBytes, ensureByteLength, lazyAsync, loadShaderParts } from "./runtime_common.js";

function clonePoint(point: CurveGPUJacobianPoint): CurveGPUJacobianPoint {
  return { x: cloneBytes(point.x), y: cloneBytes(point.y), z: cloneBytes(point.z) };
}

function unpackJacobianPoints(bytes: Uint8Array, count: number, coordinateBytes: number, pointBytes: number): CurveGPUJacobianPoint[] {
  const out: CurveGPUJacobianPoint[] = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * pointBytes;
    out.push({
      x: cloneBytes(bytes.slice(base, base + coordinateBytes)),
      y: cloneBytes(bytes.slice(base + coordinateBytes, base + 2 * coordinateBytes)),
      z: cloneBytes(bytes.slice(base + 2 * coordinateBytes, base + 3 * coordinateBytes)),
    });
  }
  return out;
}

function packAffineBases(
  bases: readonly CurveGPUAffinePoint[],
  coordinateBytes: number,
  pointBytes: number,
  oneMontZ: Uint8Array,
): Uint8Array {
  const out = new Uint8Array(bases.length * pointBytes);
  bases.forEach((base, index) => {
    ensureByteLength(base.x, coordinateBytes, `bases[${index}].x`);
    ensureByteLength(base.y, coordinateBytes, `bases[${index}].y`);
    const isInfinity = base.x.every((byte) => byte === 0) && base.y.every((byte) => byte === 0);
    const offset = index * pointBytes;
    out.set(base.x, offset);
    out.set(base.y, offset + coordinateBytes);
    out.set(isInfinity ? new Uint8Array(coordinateBytes) : oneMontZ, offset + 2 * coordinateBytes);
  });
  return out;
}

function packScalarWords(scalars: readonly Uint8Array[]): Uint32Array {
  const out = new Uint32Array(scalars.length * 8);
  scalars.forEach((scalar, index) => {
    ensureByteLength(scalar, 32, `scalars[${index}]`);
    out.set(splitBytesLEToU32(scalar), index * 8);
  });
  return out;
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
  const count = scalarsPacked.byteLength / 32;
  const out = new Uint32Array(count * 8);
  const view = new DataView(scalarsPacked.buffer, scalarsPacked.byteOffset, scalarsPacked.byteLength);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = view.getUint32(i * 4, true);
  }
  return out;
}

export function createMSMModule(
  context: CurveGPUContext,
  options: {
    curve: SupportedCurveID;
    coordinateBytes: number;
    pointBytes: number;
    shaderParts: readonly string[];
    strategy: PippengerStrategy;
  },
  fp: FieldModule,
): MSMModule {
  const { curve, coordinateBytes, pointBytes, shaderParts, strategy } = options;
  const label = `${curve}-g1-msm`;

  const getOneMontgomery = lazyAsync(async () => fp.montOne());
  const getRuntime = lazyAsync(async (): Promise<PippengerRuntime> => {
    const shaderCode = await loadShaderParts(shaderParts);
    return strategy.createRuntime(context.device, shaderCode, label);
  });

  async function runBatch(
    bases: readonly CurveGPUAffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
    options: CurveGPUMSMOptions = {},
  ): Promise<CurveGPUJacobianPoint[]> {
    if (bases.length !== scalars.length) {
      throw new Error(`${label}: bases and scalars length mismatch`);
    }
    const count = options.count ?? 1;
    const termsPerInstance = options.termsPerInstance ?? (count === 1 ? bases.length : 0);
    if (!Number.isInteger(count) || count <= 0) {
      throw new Error(`${label}: count must be a positive integer`);
    }
    if (!Number.isInteger(termsPerInstance) || termsPerInstance <= 0) {
      throw new Error(`${label}: termsPerInstance must be a positive integer`);
    }
    if (bases.length !== count * termsPerInstance) {
      throw new Error(`${label}: expected ${count * termsPerInstance} bases/scalars for count=${count} termsPerInstance=${termsPerInstance}`);
    }

    const oneMontZ = await getOneMontgomery();
    const runtime = await getRuntime();
    const window = options.window ?? bestPippengerWindow(termsPerInstance);
    const output = await runSparseSignedPippengerMSM({
      device: context.device,
      runtime,
      basesBytes: packAffineBases(bases, coordinateBytes, pointBytes, oneMontZ),
      pointBytes,
      uniformBytes: 32,
      zeroPointBytes: new Uint8Array(pointBytes),
      scalarWords: packScalarWords(scalars),
      count,
      termsPerInstance,
      window,
      maxChunkSize: options.maxChunkSize,
      labelPrefix: label,
    });
    return unpackJacobianPoints(output, count, coordinateBytes, pointBytes).map(clonePoint);
  }

  return {
    context,
    curve,
    group: "g1",
    bestWindow(termCount: number): number {
      return bestPippengerWindow(termCount);
    },
    async pippengerAffine(
      bases: readonly CurveGPUAffinePoint[],
      scalars: readonly CurveGPUElementBytes[],
      options: CurveGPUMSMOptions = {},
    ): Promise<CurveGPUJacobianPoint> {
      return (await runBatch(bases, scalars, { ...options, count: options.count ?? 1 }))[0];
    },
    async pippengerAffineBatch(
      bases: readonly CurveGPUAffinePoint[],
      scalars: readonly CurveGPUElementBytes[],
      options: CurveGPUMSMOptions,
    ): Promise<CurveGPUJacobianPoint[]> {
      return runBatch(bases, scalars, options);
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
