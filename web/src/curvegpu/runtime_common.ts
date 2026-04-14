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

function logComputePipelineCreation(label: string, entryPoint: string): void {
  const key = "__curvegpuComputePipelineCreateCount";
  const state = globalThis as typeof globalThis & { [key: string]: number | undefined };
  const count = (state[key] ?? 0) + 1;
  state[key] = count;
  console.debug(`[curvegpu] createComputePipeline #${count}: ${label} entry=${entryPoint}`);
}

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

export function createSimpleStorageBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, size),
    usage,
  });
}

export function createSimpleStorageBufferFromBytes(
  device: GPUDevice,
  label: string,
  bytes: Uint8Array,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
): GPUBuffer {
  const buffer = createSimpleStorageBuffer(device, label, bytes.byteLength, usage);
  if (bytes.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
  return buffer;
}

export function createSimpleUniformBuffer(
  device: GPUDevice,
  label: string,
  uniformWords: Uint32Array,
): GPUBuffer {
  const buffer = createSimpleStorageBuffer(
    device,
    label,
    uniformWords.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  device.queue.writeBuffer(buffer, 0, uniformWords.buffer, uniformWords.byteOffset, uniformWords.byteLength);
  return buffer;
}

export function createSimpleBindGroup(
  device: GPUDevice,
  kernel: SimpleKernel,
  label: string,
  inputA: GPUBuffer,
  inputB: GPUBuffer,
  output: GPUBuffer,
  uniform: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputA } },
      { binding: 1, resource: { buffer: inputB } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: uniform } },
    ],
  });
}

export async function submitSimpleKernel(
  device: GPUDevice,
  kernel: SimpleKernel,
  bindGroup: GPUBindGroup,
  workgroups: number,
  label: string,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  const pass = encoder.beginComputePass({ label: `${label}-pass` });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups, 1, 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

export async function readbackSimpleBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  outputBytes: number,
  label: string,
): Promise<Uint8Array> {
  let mapped = false;
  const readbackBuffer = createSimpleStorageBuffer(
    device,
    `${label}-readback`,
    outputBytes,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  );
  try {
    const encoder = device.createCommandEncoder({ label: `${label}-readback-encoder` });
    encoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, Math.max(4, outputBytes));
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
    readbackBuffer.destroy();
  }
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
  logComputePipelineCreation(`${label}-${entryPoint}`, entryPoint);
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
  const inputABuffer = createSimpleStorageBufferFromBytes(
    device,
    `${label}-input-a`,
    inputA,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const inputBBuffer = createSimpleStorageBufferFromBytes(
    device,
    `${label}-input-b`,
    inputB,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const outputBuffer = createSimpleStorageBuffer(
    device,
    `${label}-output`,
    outputBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uniformBuffer = createSimpleUniformBuffer(device, `${label}-params`, uniformWords);

  try {
    const bindGroup = createSimpleBindGroup(device, kernel, `${label}-bg`, inputABuffer, inputBBuffer, outputBuffer, uniformBuffer);
    await submitSimpleKernel(device, kernel, bindGroup, workgroups, label);
    return await readbackSimpleBuffer(device, outputBuffer, outputBytes, label);
  } finally {
    inputABuffer.destroy();
    inputBBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
  }
}
