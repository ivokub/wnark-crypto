import type { CurveGPUContext, CurveGPUElementBytes, FieldModule, NTTModule, SupportedCurveID } from "./api.js";
import {
  cloneBytes,
  createSimpleKernel,
  ensureByteLength,
  lazyAsync,
  loadShaderText,
  packElementBatch,
  runSimpleKernel,
  unpackElementBatch,
} from "./runtime_common.js";
import { fetchJSON } from "./browser_utils.js";

const VECTOR_OP_MUL_FACTORS = 3;
const VECTOR_OP_BIT_REVERSE_COPY = 4;
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
    vectorShaderPath: string;
    nttShaderPath: string;
    domainPath: string;
    modulusHex: string;
  },
  fr: FieldModule,
): NTTModule {
  const { curve, vectorShaderPath, nttShaderPath, domainPath, modulusHex } = options;
  const label = `${curve}-fr-ntt`;
  const elementBytes = fr.byteSize;
  const zeroElement = new Uint8Array(elementBytes);

  const getVectorKernel = lazyAsync(async () => {
    const shaderCode = await loadShaderText(vectorShaderPath);
    return createSimpleKernel(context.device, `${label}-vector`, shaderCode, "fr_vector_main");
  });
  const getNTTKernel = lazyAsync(async () => {
    const shaderCode = await loadShaderText(nttShaderPath);
    return createSimpleKernel(context.device, `${label}-stage`, shaderCode, "fr_ntt_stage_main");
  });
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
        forwardStageRegular.map(async (stage) => packElementBatch(await fr.toMontgomeryBatch(stage), elementBytes, `${label}.forwardStage`)),
      );
      const inverseStageMont = await Promise.all(
        inverseStageRegular.map(async (stage) => packElementBatch(await fr.toMontgomeryBatch(stage), elementBytes, `${label}.inverseStage`)),
      );
      const inverseScaleMont = await fr.toMontgomery(bigIntToBytesLE(hexToBigInt(domain.cardinality_inv_hex), elementBytes));
      return { forwardStageMont, inverseStageMont, inverseScaleMont };
    })();
    domainCache.set(size, promise);
    return promise;
  }

  async function runVectorOp(opcode: number, values: readonly Uint8Array[], factors?: readonly Uint8Array[], logCount = 0): Promise<Uint8Array[]> {
    const count = values.length;
    const kernel = await getVectorKernel();
    const output = await runSimpleKernel({
      device: context.device,
      kernel,
      label: `${label}-vector-${opcode}`,
      inputA: packElementBatch(values, elementBytes, `${label}.values`),
      inputB: packElementBatch(factors ?? Array.from({ length: count }, () => zeroElement), elementBytes, `${label}.factors`),
      outputBytes: count * elementBytes,
      uniformWords: Uint32Array.from([count, opcode, logCount, 0, 0, 0, 0, 0]),
      workgroups: Math.ceil(count / 64),
    });
    return unpackElementBatch(output, elementBytes, count);
  }

  async function runStages(values: readonly Uint8Array[], stages: readonly Uint8Array[], inverse: boolean): Promise<Uint8Array[]> {
    let state = values.map(cloneBytes);
    const kernel = await getNTTKernel();
    for (let stage = 0; stage < stages.length; stage += 1) {
      const count = state.length;
      const output = await runSimpleKernel({
        device: context.device,
        kernel,
        label: `${label}-stage-${stage}-${inverse ? "inv" : "fwd"}`,
        inputA: packElementBatch(state, elementBytes, `${label}.state`),
        inputB: cloneBytes(stages[stage]),
        outputBytes: count * elementBytes,
        uniformWords: Uint32Array.from([count, 1 << stage, inverse ? 1 : 0, 0, 0, 0, 0, 0]),
        workgroups: Math.ceil(count / 64),
      });
      state = unpackElementBatch(output, elementBytes, count);
    }
    return state;
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
      const domain = await prepareDomain(size);
      const bitReversed = await runVectorOp(VECTOR_OP_BIT_REVERSE_COPY, values, undefined, Math.round(Math.log2(size)));
      return runStages(bitReversed, domain.forwardStageMont, false);
    },
    async inverse(values: readonly CurveGPUElementBytes[]): Promise<CurveGPUElementBytes[]> {
      values.forEach((value, index) => ensureByteLength(value, elementBytes, `${label}.inverse[${index}]`));
      const size = values.length;
      if (size === 0 || (size & (size - 1)) !== 0) {
        throw new Error(`${label}: NTT input length must be a non-zero power of two`);
      }
      const domain = await prepareDomain(size);
      const bitReversed = await runVectorOp(VECTOR_OP_BIT_REVERSE_COPY, values, undefined, Math.round(Math.log2(size)));
      const staged = await runStages(bitReversed, domain.inverseStageMont, true);
      return runVectorOp(
        VECTOR_OP_MUL_FACTORS,
        staged,
        Array.from({ length: size }, () => domain.inverseScaleMont),
      );
    },
  };
}
