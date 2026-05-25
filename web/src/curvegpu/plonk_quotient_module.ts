import type { CurveGPUContext, FieldModule, NTTModule, SupportedCurveID } from "./api.js";
import {
  createSimpleStorageBuffer,
  createSimpleStorageBufferFromBytes,
  createSimpleUniformBuffer,
  loadShaderParts,
  readbackSimpleBuffer,
} from "./runtime_common.js";

declare const GPUShaderStage: { COMPUTE: number };

const PLONK_QUOTIENT_BASE_DYNAMIC_VECTOR_COUNT = 5;
const PLONK_QUOTIENT_BASE_STATIC_VECTOR_COUNT = 7;
const PLONK_QUOTIENT_BLIND_COUNT = 4;
const PLONK_QUOTIENT_SCALAR_COUNT = 7;
const PLONK_QUOTIENT_WORKGROUP_SIZE = 64;

const BN254_PLONK_QUOTIENT_SHADER_PARTS = [
  "/shaders/curves/bn254/fr_arith.wgsl#section=fr_types",
  "/shaders/curves/bn254/fr_arith.wgsl#section=fr_constants",
  "/shaders/curves/bn254/fr_arith.wgsl#section=fr_core",
  "/shaders/curves/bn254/fr_plonk_quotient.wgsl",
];

type PlonkQuotientKernel = {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  workgroupSize: number;
};

export type PlonkTransformAndEvaluateQuotientCosetInput = {
  dynamicValuesPacked: Uint8Array;
  scalingPacked: Uint8Array;
  staticValuesPacked: Uint8Array;
  twiddlesPacked: Uint8Array;
  denominatorsPacked: Uint8Array;
  blindsPacked: Uint8Array;
  scalarsPacked: Uint8Array;
  elementCount: number;
  blindCoeffCount: number;
  commitmentCount: number;
  dynamicTransformCacheKey?: number;
  staticMontCacheKey?: number;
};

export type PlonkQuotientModule = {
  readonly context: CurveGPUContext;
  readonly curve: SupportedCurveID;
  transformAndEvaluateQuotientCoset(input: PlonkTransformAndEvaluateQuotientCosetInput): Promise<Uint8Array>;
  prewarmPlonkQuotientEvaluateKernel(commitmentCount?: number): Promise<void>;
};

function cloneBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes);
}

function repeatPackedVector(value: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(value.byteLength * count);
  for (let i = 0; i < count; i += 1) {
    out.set(value, i * value.byteLength);
  }
  return out;
}

export function createPlonkQuotientModule(config: {
  context: CurveGPUContext;
  curve: SupportedCurveID;
  fr: FieldModule;
  ntt: NTTModule;
}): PlonkQuotientModule {
  const { context, curve, fr, ntt } = config;
  const quotientKernels = new Map<number, PlonkQuotientKernel>();
  let dynamicTransformCache:
    | {
        key: number;
        elementCount: number;
        dynamicVectorCount: number;
        coeffMont: Uint8Array;
      }
    | null = null;
  const staticMontCache = new Map<
    number,
    {
      elementCount: number;
      staticVectorCount: number;
      mont: Uint8Array;
    }
  >();

  async function getQuotientKernel(commitmentCount: number): Promise<PlonkQuotientKernel> {
    if (curve !== "bn254") {
      throw new Error(`PLONK quotient WebGPU evaluator only supports bn254, got ${curve}`);
    }
    if (!Number.isInteger(commitmentCount) || commitmentCount < 0) {
      throw new Error(`invalid PLONK quotient commitment count ${commitmentCount}`);
    }
    const device = context.device;
    const cached = quotientKernels.get(commitmentCount);
    if (cached?.device === device) {
      return cached;
    }

    const bindGroupLayout = device.createBindGroupLayout({
      label: "plonk-bn254-quotient-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "plonk-bn254-quotient-pl",
      bindGroupLayouts: [bindGroupLayout],
    });
    const code = await loadShaderParts(BN254_PLONK_QUOTIENT_SHADER_PARTS);
    const shader = device.createShaderModule({
      label: "plonk-bn254-quotient-shader",
      code,
    });
    const pipeline = await device.createComputePipelineAsync({
      label: `plonk-bn254-quotient-c${commitmentCount}`,
      layout: pipelineLayout,
      compute: {
        module: shader,
        entryPoint: "fr_plonk_quotient_main",
        constants: {
          WORKGROUP_SIZE: PLONK_QUOTIENT_WORKGROUP_SIZE,
          COMMITMENT_COUNT: commitmentCount,
        },
      },
    });
    const kernel = {
      device,
      pipeline,
      bindGroupLayout,
      workgroupSize: PLONK_QUOTIENT_WORKGROUP_SIZE,
    };
    quotientKernels.set(commitmentCount, kernel);
    return kernel;
  }

  async function runQuotientKernelMont(
    vectorsMontPacked: Uint8Array,
    blindsMontPacked: Uint8Array,
    scalarsMontPacked: Uint8Array,
    elementCount: number,
    blindCoeffCount: number,
    commitmentCount: number,
  ) {
    const elementBytes = fr.byteSize;
    const vectorBytes = elementCount * elementBytes;
    const device = context.device;
    const kernel = await getQuotientKernel(commitmentCount);
    const vectorsBuffer = createSimpleStorageBufferFromBytes(device, "plonk-quotient-vectors", vectorsMontPacked);
    const blindsBuffer = createSimpleStorageBufferFromBytes(device, "plonk-quotient-blinds", blindsMontPacked);
    const scalarsBuffer = createSimpleStorageBufferFromBytes(device, "plonk-quotient-scalars", scalarsMontPacked);
    const outputBuffer = createSimpleStorageBuffer(device, "plonk-quotient-output", vectorBytes);
    const paramsBuffer = createSimpleUniformBuffer(
      device,
      "plonk-quotient-params",
      new Uint32Array([elementCount, blindCoeffCount, 0, 0]),
    );

    try {
      const bindGroup = device.createBindGroup({
        label: "plonk-quotient-bg",
        layout: kernel.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: vectorsBuffer } },
          { binding: 1, resource: { buffer: blindsBuffer } },
          { binding: 2, resource: { buffer: scalarsBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      });
      const encoder = device.createCommandEncoder({ label: "plonk-quotient-encoder" });
      const pass = encoder.beginComputePass({ label: "plonk-quotient-pass" });
      pass.setPipeline(kernel.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(elementCount / kernel.workgroupSize), 1, 1);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return await readbackSimpleBuffer(device, outputBuffer, vectorBytes, "plonk-quotient");
    } finally {
      vectorsBuffer.destroy();
      blindsBuffer.destroy();
      scalarsBuffer.destroy();
      outputBuffer.destroy();
      paramsBuffer.destroy();
    }
  }

  async function transformAndEvaluateQuotientCoset(input: PlonkTransformAndEvaluateQuotientCosetInput): Promise<Uint8Array> {
    const {
      dynamicValuesPacked,
      scalingPacked,
      staticValuesPacked,
      twiddlesPacked,
      denominatorsPacked,
      blindsPacked,
      scalarsPacked,
      elementCount,
      blindCoeffCount,
      commitmentCount,
      dynamicTransformCacheKey = 0,
      staticMontCacheKey = 0,
    } = input;
    const elementBytes = fr.byteSize;
    const vectorBytes = elementCount * elementBytes;
    if (!Number.isInteger(elementCount) || elementCount <= 0 || (elementCount & (elementCount - 1)) !== 0) {
      throw new Error(`invalid PLONK quotient evaluate element count ${elementCount}`);
    }
    if (!Number.isInteger(blindCoeffCount) || blindCoeffCount < 0) {
      throw new Error(`invalid PLONK quotient blind coefficient count ${blindCoeffCount}`);
    }
    if (!Number.isInteger(commitmentCount) || commitmentCount < 0) {
      throw new Error(`invalid PLONK quotient commitment count ${commitmentCount}`);
    }

    const dynamicVectorCount = PLONK_QUOTIENT_BASE_DYNAMIC_VECTOR_COUNT + commitmentCount;
    const staticVectorCount = PLONK_QUOTIENT_BASE_STATIC_VECTOR_COUNT + commitmentCount;
    const vectorCount = dynamicVectorCount + staticVectorCount + 2;
    const expectedDynamicBytes = dynamicVectorCount * vectorBytes;
    const canReuseDynamicCache =
      dynamicTransformCacheKey > 0 &&
      dynamicTransformCache?.key === dynamicTransformCacheKey &&
      dynamicTransformCache.elementCount === elementCount &&
      dynamicTransformCache.dynamicVectorCount === dynamicVectorCount;
    const cachedDynamicCoeffMont = canReuseDynamicCache ? dynamicTransformCache?.coeffMont : undefined;
    if (dynamicValuesPacked.byteLength !== expectedDynamicBytes && !(dynamicValuesPacked.byteLength === 0 && canReuseDynamicCache)) {
      throw new Error(
        `PLONK quotient transform/evaluate expected ${expectedDynamicBytes} dynamic bytes, got ${dynamicValuesPacked.byteLength}`,
      );
    }
    if (scalingPacked.byteLength !== vectorBytes) {
      throw new Error(`PLONK quotient transform/evaluate expected ${vectorBytes} scaling bytes, got ${scalingPacked.byteLength}`);
    }
    const expectedStaticBytes = staticVectorCount * vectorBytes;
    const cachedStatic = staticMontCacheKey > 0 ? staticMontCache.get(staticMontCacheKey) : undefined;
    const canReuseStaticCache =
      staticMontCacheKey > 0 &&
      cachedStatic?.elementCount === elementCount &&
      cachedStatic.staticVectorCount === staticVectorCount;
    if (staticValuesPacked.byteLength !== expectedStaticBytes && !(staticValuesPacked.byteLength === 0 && canReuseStaticCache)) {
      throw new Error(
        `PLONK quotient transform/evaluate expected ${expectedStaticBytes} static bytes, got ${staticValuesPacked.byteLength}`,
      );
    }
    if (twiddlesPacked.byteLength !== vectorBytes) {
      throw new Error(`PLONK quotient transform/evaluate expected ${vectorBytes} twiddle bytes, got ${twiddlesPacked.byteLength}`);
    }
    if (denominatorsPacked.byteLength !== vectorBytes) {
      throw new Error(`PLONK quotient transform/evaluate expected ${vectorBytes} denominator bytes, got ${denominatorsPacked.byteLength}`);
    }
    const blindBytes = PLONK_QUOTIENT_BLIND_COUNT * blindCoeffCount * elementBytes;
    if (blindsPacked.byteLength !== blindBytes) {
      throw new Error(`PLONK quotient transform/evaluate expected ${blindBytes} blinding bytes, got ${blindsPacked.byteLength}`);
    }
    const scalarBytes = PLONK_QUOTIENT_SCALAR_COUNT * elementBytes;
    if (scalarsPacked.byteLength !== scalarBytes) {
      throw new Error(`PLONK quotient transform/evaluate expected ${scalarBytes} scalar bytes, got ${scalarsPacked.byteLength}`);
    }

    const vectorsMontPacked = new Uint8Array(vectorCount * vectorBytes);
    const dynamicCoeffMont =
      cachedDynamicCoeffMont
        ? cachedDynamicCoeffMont
        : await (async (): Promise<Uint8Array> => {
            const dynamicMont = await fr.toMontgomeryPacked(cloneBytes(dynamicValuesPacked));
            const coeffMont = await ntt.inversePackedMontBatch(dynamicMont, elementCount, dynamicVectorCount);
            if (dynamicTransformCacheKey > 0) {
              dynamicTransformCache = {
                key: dynamicTransformCacheKey,
                elementCount,
                dynamicVectorCount,
                coeffMont,
              };
            }
            return coeffMont;
          })();
    const scalingMont = await fr.toMontgomeryPacked(cloneBytes(scalingPacked));
    const scalingMontBatch = repeatPackedVector(scalingMont, dynamicVectorCount);
    const shiftedCoeffMont = await fr.mulPackedMont(dynamicCoeffMont, scalingMontBatch);
    vectorsMontPacked.set(await ntt.forwardPackedMontBatch(shiftedCoeffMont, elementCount, dynamicVectorCount));

    const cachedStaticMont = canReuseStaticCache ? cachedStatic?.mont : undefined;
    const staticMontPromise = cachedStaticMont
      ? Promise.resolve(cachedStaticMont)
      : fr.toMontgomeryPacked(cloneBytes(staticValuesPacked)).then((mont) => {
          if (staticMontCacheKey > 0) {
            staticMontCache.set(staticMontCacheKey, {
              elementCount,
              staticVectorCount,
              mont,
            });
          }
          return mont;
        });
    const [staticMont, twiddlesMont, denominatorsMont, blindsMont, scalarsMont] = await Promise.all([
      staticMontPromise,
      fr.toMontgomeryPacked(cloneBytes(twiddlesPacked)),
      fr.toMontgomeryPacked(cloneBytes(denominatorsPacked)),
      fr.toMontgomeryPacked(cloneBytes(blindsPacked)),
      fr.toMontgomeryPacked(cloneBytes(scalarsPacked)),
    ]);

    vectorsMontPacked.set(staticMont, dynamicVectorCount * vectorBytes);
    vectorsMontPacked.set(twiddlesMont, (dynamicVectorCount + staticVectorCount) * vectorBytes);
    vectorsMontPacked.set(denominatorsMont, (dynamicVectorCount + staticVectorCount + 1) * vectorBytes);
    return runQuotientKernelMont(vectorsMontPacked, blindsMont, scalarsMont, elementCount, blindCoeffCount, commitmentCount);
  }

  return {
    context,
    curve,
    transformAndEvaluateQuotientCoset,
    async prewarmPlonkQuotientEvaluateKernel(commitmentCount = 0): Promise<void> {
      await getQuotientKernel(commitmentCount);
    },
  };
}
