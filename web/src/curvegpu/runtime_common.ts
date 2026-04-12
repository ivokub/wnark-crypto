import { fetchText } from "./browser_utils.js";
import { fetchShaderParts } from "./shaders.js";

declare const GPUShaderStage: { COMPUTE: number };
declare const GPUBufferUsage: {
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  MAP_READ: number;
  UNIFORM: number;
};
declare const GPUMapMode: { READ: number };

export type SimpleKernel = {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
};

export function lazyAsync<T>(factory: () => Promise<T>): () => Promise<T> {
  let promise: Promise<T> | null = null;
  return (): Promise<T> => {
    if (!promise) {
      promise = factory();
    }
    return promise;
  };
}

export function cloneBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes);
}

export function ensureByteLength(bytes: Uint8Array, expected: number, label: string): void {
  if (bytes.byteLength !== expected) {
    throw new Error(`${label}: expected ${expected} bytes, got ${bytes.byteLength}`);
  }
}

export function packElementBatch(values: readonly Uint8Array[], elementBytes: number, label: string): Uint8Array {
  const out = new Uint8Array(values.length * elementBytes);
  values.forEach((value, index) => {
    ensureByteLength(value, elementBytes, `${label}[${index}]`);
    out.set(value, index * elementBytes);
  });
  return out;
}

export function unpackElementBatch(bytes: Uint8Array, elementBytes: number, count: number): Uint8Array[] {
  const out: Uint8Array[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(cloneBytes(bytes.slice(i * elementBytes, (i + 1) * elementBytes)));
  }
  return out;
}

export async function loadShaderText(path: string): Promise<string> {
  return fetchText(path);
}

export async function loadShaderParts(parts: readonly string[]): Promise<string> {
  return fetchShaderParts(parts);
}

export function createSimpleKernel(
  device: GPUDevice,
  label: string,
  shaderCode: string,
  entryPoint: string,
): SimpleKernel {
  const shaderModule = device.createShaderModule({
    label: `${label}-shader`,
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${label}-pl`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: `${label}-${entryPoint}`,
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

export async function runSimpleKernel(options: {
  device: GPUDevice;
  kernel: SimpleKernel;
  label: string;
  inputA: Uint8Array;
  inputB: Uint8Array;
  outputBytes: number;
  uniformWords: Uint32Array;
  workgroups: number;
}): Promise<Uint8Array> {
  const { device, kernel, label, inputA, inputB, outputBytes, uniformWords, workgroups } = options;
  let mapped = false;
  const inputABuffer = createStorageBuffer(
    device,
    `${label}-input-a`,
    Math.max(4, inputA.byteLength),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const inputBBuffer = createStorageBuffer(
    device,
    `${label}-input-b`,
    Math.max(4, inputB.byteLength),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const outputBuffer = createStorageBuffer(
    device,
    `${label}-output`,
    Math.max(4, outputBytes),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const readbackBuffer = createStorageBuffer(
    device,
    `${label}-readback`,
    Math.max(4, outputBytes),
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  );
  const uniformBuffer = createStorageBuffer(
    device,
    `${label}-params`,
    Math.max(4, uniformWords.byteLength),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );

  try {
    if (inputA.byteLength > 0) {
      device.queue.writeBuffer(inputABuffer, 0, inputA.buffer.slice(inputA.byteOffset, inputA.byteOffset + inputA.byteLength));
    }
    if (inputB.byteLength > 0) {
      device.queue.writeBuffer(inputBBuffer, 0, inputB.buffer.slice(inputB.byteOffset, inputB.byteOffset + inputB.byteLength));
    }
    device.queue.writeBuffer(uniformBuffer, 0, uniformWords.buffer, uniformWords.byteOffset, uniformWords.byteLength);

    const bindGroup = device.createBindGroup({
      label: `${label}-bg`,
      layout: kernel.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputABuffer } },
        { binding: 1, resource: { buffer: inputBBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
    const pass = encoder.beginComputePass({ label: `${label}-pass` });
    pass.setPipeline(kernel.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups, 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, Math.max(4, outputBytes));
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    const range = readbackBuffer.getMappedRange();
    const out = new Uint8Array(range.slice(0, outputBytes));
    readbackBuffer.unmap();
    mapped = false;
    return out;
  } finally {
    if (mapped) {
      readbackBuffer.unmap();
    }
    inputABuffer.destroy();
    inputBBuffer.destroy();
    outputBuffer.destroy();
    readbackBuffer.destroy();
    uniformBuffer.destroy();
  }
}
