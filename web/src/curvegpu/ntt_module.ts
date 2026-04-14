import type { CurveGPUContext, CurveGPUElementBytes, FieldModule, NTTModule, SupportedCurveID } from "./api.js";
import type { SimpleKernel } from "./runtime_common.js";
import {
  cloneBytes,
  createSimpleBindGroup,
  createSimpleStorageBuffer,
  createSimpleStorageBufferFromBytes,
  createSimpleUniformBuffer,
  ensureByteLength,
  lazyAsync,
  packElementBatch,
  readbackSimpleBuffer,
  runSimpleKernel,
  submitSimpleKernel,
  unpackElementBatch,
} from "./runtime_common.js";
import { fetchJSON } from "./browser_utils.js";

declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
};

const VECTOR_OP_MUL_FACTORS = 3;
const VECTOR_OP_BIT_REVERSE_COPY = 4;
const FIELD_OP_TO_MONT = 11;
const FIELD_OP_FROM_MONT = 12;
const UNIFORM_WORDS = 8;
type DomainMetadata = {
  log_n: number;
  size: number;
  omega_hex: string;
  omega_inv_hex: string;
  cardinality_inv_hex: string;
};

type DomainMetadataFile = {
  domains: DomainMetadata[];
};

type PreparedDomain = {
  forwardStageMont: Uint8Array[];
  inverseStageMont: Uint8Array[];
  inverseScaleMont: Uint8Array;
  inverseScaleFactorsPackedMont: Uint8Array;
};

function hexToBigInt(hex: string): bigint {
  return BigInt(`0x${hex}`);
}

function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  let acc = base % mod;
  let power = exp;
  while (power > 0n) {
    if ((power & 1n) === 1n) {
      result = (result * acc) % mod;
    }
    acc = (acc * acc) % mod;
    power >>= 1n;
  }
  return result;
}

function bigIntToBytesLE(value: bigint, byteSize: number): Uint8Array {
  const out = new Uint8Array(byteSize);
  let x = value;
  for (let i = 0; i < byteSize; i += 1) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return out;
}

function ensurePackedElements(bytes: Uint8Array, elementBytes: number, label: string): number {
  if (bytes.byteLength % elementBytes !== 0) {
    throw new Error(`${label}: expected a multiple of ${elementBytes} bytes, got ${bytes.byteLength}`);
  }
  return bytes.byteLength / elementBytes;
}

function repeatPackedElement(value: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(value.byteLength * count);
  for (let i = 0; i < count; i += 1) {
    out.set(value, i * value.byteLength);
  }
  return out;
}

function buildRegularStageElements(domain: DomainMetadata, inverse: boolean, modulus: bigint, elementBytes: number): Uint8Array[][] {
  const logN = domain.log_n;
  const omega = hexToBigInt(inverse ? domain.omega_inv_hex : domain.omega_hex);
  const stages: Uint8Array[][] = [];
  for (let stage = 1; stage <= logN; stage += 1) {
    const m = 1 << (stage - 1);
    const exponentShift = BigInt(logN - stage);
    const step = modPow(omega, 1n << exponentShift, modulus);
    const stageElements: Uint8Array[] = [];
    let acc = 1n;
    for (let i = 0; i < m; i += 1) {
      stageElements.push(bigIntToBytesLE(acc, elementBytes));
      acc = (acc * step) % modulus;
    }
    stages.push(stageElements);
  }
  return stages;
}

export function createNTTModule(
  context: CurveGPUContext,
  options: {
    curve: SupportedCurveID;
    vectorKernel: SimpleKernel;
    fieldKernel: SimpleKernel;
    nttKernel: SimpleKernel;
    domainPath: string;
    modulusHex: string;
  },
  fr: FieldModule,
): NTTModule {
  const { curve, vectorKernel, fieldKernel, nttKernel, domainPath, modulusHex } = options;
  const label = `${curve}-fr-ntt`;
  const elementBytes = fr.byteSize;
  const zeroElement = new Uint8Array(elementBytes);

  const getVectorKernel = lazyAsync(async () => vectorKernel);
  const getFieldKernel = lazyAsync(async () => fieldKernel);
  const getNTTKernel = lazyAsync(async () => nttKernel);
  const getDomains = lazyAsync(async () => fetchJSON<DomainMetadataFile>(domainPath));
  const domainCache = new Map<number, Promise<PreparedDomain>>();
  const modulus = BigInt(modulusHex);

  async function prepareDomain(size: number): Promise<PreparedDomain> {
    const cached = domainCache.get(size);
    if (cached) {
      return cached;
    }
    const promise = (async (): Promise<PreparedDomain> => {
      const file = await getDomains();
      const domain = file.domains.find((item) => item.size === size);
      if (!domain) {
        throw new Error(`${label}: missing domain metadata for size ${size}`);
      }
      const forwardStageRegular = buildRegularStageElements(domain, false, modulus, elementBytes);
      const inverseStageRegular = buildRegularStageElements(domain, true, modulus, elementBytes);
      const forwardStageMont = await Promise.all(
        forwardStageRegular.map(async (stage) =>
          fr.toMontgomeryPacked(packElementBatch(stage, elementBytes, `${label}.forwardStageRegular`)),
        ),
      );
      const inverseStageMont = await Promise.all(
        inverseStageRegular.map(async (stage) =>
          fr.toMontgomeryPacked(packElementBatch(stage, elementBytes, `${label}.inverseStageRegular`)),
        ),
      );
      const inverseScaleMont = await fr.toMontgomery(bigIntToBytesLE(hexToBigInt(domain.cardinality_inv_hex), elementBytes));
      return {
        forwardStageMont,
        inverseStageMont,
        inverseScaleMont,
        inverseScaleFactorsPackedMont: repeatPackedElement(inverseScaleMont, size),
      };
    })();
    domainCache.set(size, promise);
    return promise;
  }

  async function runVectorOp(opcode: number, values: readonly Uint8Array[], factors?: readonly Uint8Array[], logCount = 0): Promise<Uint8Array[]> {
    const count = values.length;
    const kernel = await getVectorKernel();
    const output = await runSimpleKernel({
      device: context.device,
      pool: context.bufferPool,
      kernel,
      label: `${label}-vector-${opcode}`,
      inputA: packElementBatch(values, elementBytes, `${label}.values`),
      inputB: packElementBatch(factors ?? Array.from({ length: count }, () => zeroElement), elementBytes, `${label}.factors`),
      outputBytes: count * elementBytes,
      uniformWords: Uint32Array.from([count, opcode, logCount, 0, 0, 0, 0, 0]),
      workgroups: Math.ceil(count / kernel.workgroupSize),
    });
    return unpackElementBatch(output, elementBytes, count);
  }

  async function runVectorOpPacked(opcode: number, valuesPacked: Uint8Array, factorsPacked?: Uint8Array, logCount = 0): Promise<Uint8Array> {
    const count = ensurePackedElements(valuesPacked, elementBytes, `${label}.valuesPacked`);
    const kernel = await getVectorKernel();
    const factorBytes = factorsPacked ?? new Uint8Array(valuesPacked.byteLength);
    if (factorBytes.byteLength !== valuesPacked.byteLength) {
      throw new Error(`${label}.factorsPacked: expected ${valuesPacked.byteLength} bytes, got ${factorBytes.byteLength}`);
    }
    return runSimpleKernel({
      device: context.device,
      pool: context.bufferPool,
      kernel,
      label: `${label}-vector-packed-${opcode}`,
      inputA: valuesPacked,
      inputB: factorBytes,
      outputBytes: count * elementBytes,
      uniformWords: Uint32Array.from([count, opcode, logCount, 0, 0, 0, 0, 0]),
      workgroups: Math.ceil(count / kernel.workgroupSize),
    });
  }

  async function runStages(values: readonly Uint8Array[], stages: readonly Uint8Array[], inverse: boolean): Promise<Uint8Array[]> {
    let state = values.map(cloneBytes);
    const kernel = await getNTTKernel();
    for (let stage = 0; stage < stages.length; stage += 1) {
      const count = state.length;
      const output = await runSimpleKernel({
        device: context.device,
        pool: context.bufferPool,
        kernel,
        label: `${label}-stage-${stage}-${inverse ? "inv" : "fwd"}`,
        inputA: packElementBatch(state, elementBytes, `${label}.state`),
        inputB: cloneBytes(stages[stage]),
        outputBytes: count * elementBytes,
        uniformWords: Uint32Array.from([count, 1 << stage, inverse ? 1 : 0, 0, 0, 0, 0, 0]),
        workgroups: Math.ceil(count / kernel.workgroupSize),
      });
      state = unpackElementBatch(output, elementBytes, count);
    }
    return state;
  }

  async function runStagesPacked(valuesPacked: Uint8Array, stages: readonly Uint8Array[], inverse: boolean): Promise<Uint8Array> {
    let state = cloneBytes(valuesPacked);
    const kernel = await getNTTKernel();
    const count = ensurePackedElements(state, elementBytes, `${label}.statePacked`);
    for (let stage = 0; stage < stages.length; stage += 1) {
      state = await runSimpleKernel({
        device: context.device,
        pool: context.bufferPool,
        kernel,
        label: `${label}-stage-packed-${stage}-${inverse ? "inv" : "fwd"}`,
        inputA: state,
        inputB: cloneBytes(stages[stage]),
        outputBytes: count * elementBytes,
        uniformWords: Uint32Array.from([count, 1 << stage, inverse ? 1 : 0, 0, 0, 0, 0, 0]),
        workgroups: Math.ceil(count / kernel.workgroupSize),
      });
    }
    return state;
  }

  async function runPipelinePacked(options: {
    values: Uint8Array;
    inverse: boolean;
    inputRegular: boolean;
    outputRegular: boolean;
  }): Promise<Uint8Array> {
    const { values, inverse, inputRegular, outputRegular } = options;
    const count = ensurePackedElements(values, elementBytes, `${label}.pipeline.values`);
    if (count === 0 || (count & (count - 1)) !== 0) {
      throw new Error(`${label}: NTT input length must be a non-zero power of two`);
    }

    const totalBytes = count * elementBytes;
    const domain = await prepareDomain(count);
    const [fieldKernel, vectorKernel, nttKernel] = await Promise.all([getFieldKernel(), getVectorKernel(), getNTTKernel()]);
    const zeroAux = createSimpleStorageBufferFromBytes(
      context.device,
      `${label}-zero-aux`,
      new Uint8Array(4),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    let current = createSimpleStorageBufferFromBytes(
      context.device,
      `${label}-state-a`,
      values,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    );
    let next = createSimpleStorageBuffer(
      context.device,
      `${label}-state-b`,
      totalBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    );

    const dispatch = async (
      kernel: SimpleKernel,
      inputA: GPUBuffer,
      inputB: GPUBuffer,
      output: GPUBuffer,
      uniformWords: Uint32Array,
      opLabel: string,
    ): Promise<void> => {
      const uniform = createSimpleUniformBuffer(context.device, `${opLabel}-params`, uniformWords);
      try {
        const bindGroup = createSimpleBindGroup(context.device, kernel, `${opLabel}-bg`, inputA, inputB, output, uniform);
        await submitSimpleKernel(context.device, kernel, bindGroup, Math.ceil(count / kernel.workgroupSize), opLabel);
      } finally {
        uniform.destroy();
      }
    };

    const swap = (): void => {
      const tmp = current;
      current = next;
      next = tmp;
    };

    try {
      if (inputRegular) {
        await dispatch(
          fieldKernel,
          current,
          zeroAux,
          next,
          Uint32Array.from([count, FIELD_OP_TO_MONT, 0, 0, 0, 0, 0, 0]),
          `${label}-to-mont`,
        );
        swap();
      }

      await dispatch(
        vectorKernel,
        current,
        zeroAux,
        next,
        Uint32Array.from([count, VECTOR_OP_BIT_REVERSE_COPY, Math.round(Math.log2(count)), 0, 0, 0, 0, 0]),
        `${label}-bit-reverse`,
      );
      swap();

      const stages = inverse ? domain.inverseStageMont : domain.forwardStageMont;
      for (let stage = 0; stage < stages.length; stage += 1) {
        const twiddleBuffer = createSimpleStorageBufferFromBytes(
          context.device,
          `${label}-twiddles-${stage}`,
          stages[stage],
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        );
        try {
          await dispatch(
            nttKernel,
            current,
            twiddleBuffer,
            next,
            Uint32Array.from([count, 1 << stage, inverse ? 1 : 0, 0, 0, 0, 0, 0]),
            `${label}-stage-${stage}-${inverse ? "inv" : "fwd"}`,
          );
        } finally {
          twiddleBuffer.destroy();
        }
        swap();
      }

      if (inverse) {
        const factorBuffer = createSimpleStorageBufferFromBytes(
          context.device,
          `${label}-inverse-scale`,
          domain.inverseScaleFactorsPackedMont,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        );
        try {
          await dispatch(
            vectorKernel,
            current,
            factorBuffer,
            next,
            Uint32Array.from([count, VECTOR_OP_MUL_FACTORS, 0, 0, 0, 0, 0, 0]),
            `${label}-inverse-scale`,
          );
        } finally {
          factorBuffer.destroy();
        }
        swap();
      }

      if (outputRegular) {
        await dispatch(
          fieldKernel,
          current,
          zeroAux,
          next,
          Uint32Array.from([count, FIELD_OP_FROM_MONT, 0, 0, 0, 0, 0, 0]),
          `${label}-from-mont`,
        );
        swap();
      }

      return await readbackSimpleBuffer(context.device, current, totalBytes, `${label}-pipeline`);
    } finally {
      zeroAux.destroy();
      current.destroy();
      next.destroy();
    }
  }

  return {
    context,
    curve,
    field: "fr",
    async supportedSizes(): Promise<number[]> {
      const file = await getDomains();
      return file.domains.map((domain) => domain.size).sort((a, b) => a - b);
    },
    async forward(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]> {
      values.forEach((value, index) => ensureByteLength(value, elementBytes, `${label}.forward[${index}]`));
      const size = values.length;
      if (size === 0 || (size & (size - 1)) !== 0) {
        throw new Error(`${label}: NTT input length must be a non-zero power of two`);
      }
      const output = await runPipelinePacked({
        values: packElementBatch(values, elementBytes, `${label}.forward.values`),
        inverse: false,
        inputRegular: false,
        outputRegular: false,
      });
      return unpackElementBatch(output, elementBytes, size);
    },
    async inverse(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]> {
      values.forEach((value, index) => ensureByteLength(value, elementBytes, `${label}.inverse[${index}]`));
      const size = values.length;
      if (size === 0 || (size & (size - 1)) !== 0) {
        throw new Error(`${label}: NTT input length must be a non-zero power of two`);
      }
      const output = await runPipelinePacked({
        values: packElementBatch(values, elementBytes, `${label}.inverse.values`),
        inverse: true,
        inputRegular: false,
        outputRegular: false,
      });
      return unpackElementBatch(output, elementBytes, size);
    },
    async forwardPackedMont(values: Uint8Array): Promise<Uint8Array> {
      return runPipelinePacked({ values, inverse: false, inputRegular: false, outputRegular: false });
    },
    async inversePackedMont(values: Uint8Array): Promise<Uint8Array> {
      return runPipelinePacked({ values, inverse: true, inputRegular: false, outputRegular: false });
    },
    async forwardPackedRegular(values: Uint8Array): Promise<Uint8Array> {
      return runPipelinePacked({ values, inverse: false, inputRegular: true, outputRegular: true });
    },
    async inversePackedRegular(values: Uint8Array): Promise<Uint8Array> {
      return runPipelinePacked({ values, inverse: true, inputRegular: true, outputRegular: true });
    },
  };
}
