import type {
  CurveGPUContext,
  CurveGPUElementBytes,
  CurveGPUG2AffinePoint,
  CurveGPUG2JacobianPoint,
  CurveGPUMSMOptions,
  G2Module,
  G2MSMModule,
  SupportedCurveID,
} from "./api.js";
import { bestPippengerWindow } from "./msm_shared.js";

function isZeroBytes(bytes: Uint8Array): boolean {
  return bytes.every((byte) => byte === 0);
}

function isAffineInfinity(point: CurveGPUG2AffinePoint): boolean {
  return isZeroBytes(point.x.c0) && isZeroBytes(point.x.c1) && isZeroBytes(point.y.c0) && isZeroBytes(point.y.c1);
}

function isJacobianInfinity(point: CurveGPUG2JacobianPoint): boolean {
  return isZeroBytes(point.z.c0) && isZeroBytes(point.z.c1);
}

function cloneFp2(point: CurveGPUG2AffinePoint["x"]): CurveGPUG2AffinePoint["x"] {
  return {
    c0: new Uint8Array(point.c0),
    c1: new Uint8Array(point.c1),
  };
}

function cloneAffine(point: CurveGPUG2AffinePoint): CurveGPUG2AffinePoint {
  return {
    x: cloneFp2(point.x),
    y: cloneFp2(point.y),
  };
}

export function createG2MSMModule(
  context: CurveGPUContext,
  options: {
    curve: SupportedCurveID;
  },
  g2: G2Module,
): G2MSMModule {
  const { curve } = options;
  const label = `${curve}-g2-msm`;

  async function runInstance(
    bases: readonly CurveGPUG2AffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
  ): Promise<CurveGPUG2JacobianPoint> {
    const scaled = await g2.scalarMulAffineBatch(bases, scalars);
    let acc = g2.jacobianZero();
    for (const point of scaled) {
      const affine = cloneAffine({ x: point.x, y: point.y });
      if (!isAffineInfinity(affine)) {
        acc = await g2.addMixed(acc, affine);
      }
    }
    if (isJacobianInfinity(acc)) {
      return acc;
    }
    return g2.affineToJacobian(await g2.jacobianToAffine(acc));
  }

  async function runBatch(
    bases: readonly CurveGPUG2AffinePoint[],
    scalars: readonly CurveGPUElementBytes[],
    options: CurveGPUMSMOptions = {},
  ): Promise<CurveGPUG2JacobianPoint[]> {
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

    const out = new Array<CurveGPUG2JacobianPoint>(count);
    for (let instance = 0; instance < count; instance += 1) {
      const start = instance * termsPerInstance;
      const instanceBases = bases.slice(start, start + termsPerInstance);
      const instanceScalars = scalars.slice(start, start + termsPerInstance);
      out[instance] = await runInstance(instanceBases, instanceScalars);
    }
    return out;
  }

  function unpackPackedBases(
    basesPacked: Uint8Array,
    count: number,
    termsPerInstance: number,
  ): CurveGPUG2AffinePoint[] {
    const pointBytes = g2.pointBytes;
    const componentBytes = g2.componentBytes;
    const expected = count * termsPerInstance * pointBytes;
    if (basesPacked.byteLength !== expected) {
      throw new Error(`${label}: expected ${expected} base bytes, got ${basesPacked.byteLength}`);
    }
    const out: CurveGPUG2AffinePoint[] = [];
    for (let i = 0; i < count * termsPerInstance; i += 1) {
      const base = i * pointBytes;
      out.push({
        x: {
          c0: new Uint8Array(basesPacked.slice(base, base + componentBytes)),
          c1: new Uint8Array(basesPacked.slice(base + componentBytes, base + 2 * componentBytes)),
        },
        y: {
          c0: new Uint8Array(basesPacked.slice(base + 2 * componentBytes, base + 3 * componentBytes)),
          c1: new Uint8Array(basesPacked.slice(base + 3 * componentBytes, base + 4 * componentBytes)),
        },
      });
    }
    return out;
  }

  function unpackPackedScalars(scalarsPacked: Uint8Array, count: number): Uint8Array[] {
    const expected = count * 32;
    if (scalarsPacked.byteLength !== expected) {
      throw new Error(`${label}: expected ${expected} scalar bytes, got ${scalarsPacked.byteLength}`);
    }
    const out = new Array<Uint8Array>(count);
    for (let i = 0; i < count; i += 1) {
      out[i] = new Uint8Array(scalarsPacked.slice(i * 32, (i + 1) * 32));
    }
    return out;
  }

  function packJacobianPoints(points: readonly CurveGPUG2JacobianPoint[]): Uint8Array {
    const out = new Uint8Array(points.length * g2.pointBytes);
    for (let i = 0; i < points.length; i += 1) {
      const base = i * g2.pointBytes;
      out.set(points[i].x.c0, base);
      out.set(points[i].x.c1, base + g2.componentBytes);
      out.set(points[i].y.c0, base + 2 * g2.componentBytes);
      out.set(points[i].y.c1, base + 3 * g2.componentBytes);
      out.set(points[i].z.c0, base + 4 * g2.componentBytes);
      out.set(points[i].z.c1, base + 5 * g2.componentBytes);
    }
    return out;
  }

  return {
    context,
    curve,
    group: "g2",
    bestWindow(termCount: number): number {
      return bestPippengerWindow(termCount);
    },
    async pippengerAffine(
      bases: readonly CurveGPUG2AffinePoint[],
      scalars: readonly CurveGPUElementBytes[],
      options: CurveGPUMSMOptions = {},
    ): Promise<CurveGPUG2JacobianPoint> {
      return (await runBatch(bases, scalars, { ...options, count: options.count ?? 1 }))[0];
    },
    async pippengerAffineBatch(
      bases: readonly CurveGPUG2AffinePoint[],
      scalars: readonly CurveGPUElementBytes[],
      options: CurveGPUMSMOptions,
    ): Promise<CurveGPUG2JacobianPoint[]> {
      return runBatch(bases, scalars, options);
    },
    async pippengerPackedJacobianBases(
      basesPacked: Uint8Array,
      scalarsPacked: Uint8Array,
      options: CurveGPUMSMOptions,
    ): Promise<Uint8Array> {
      const count = options.count ?? 1;
      const termsPerInstance = options.termsPerInstance ?? 0;
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`${label}: count must be a positive integer`);
      }
      if (!Number.isInteger(termsPerInstance) || termsPerInstance <= 0) {
        throw new Error(`${label}: termsPerInstance must be a positive integer`);
      }
      const bases = unpackPackedBases(basesPacked, count, termsPerInstance);
      const scalars = unpackPackedScalars(scalarsPacked, count * termsPerInstance);
      const result = await runBatch(bases, scalars, options);
      return packJacobianPoints(result);
    },
  };
}
