import { getAdapterInfo } from "./browser_utils.js";
import type {
  CurveGPUAdapterDiagnostics,
  CurveGPUContext,
  CurveGPUContextOptions,
  CurveGPURequestedLimits,
} from "./api.js";

type AdapterWithLimits = GPUAdapter & {
  isFallbackAdapter?: boolean;
  limits?: {
    maxStorageBufferBindingSize?: number;
    maxBufferSize?: number;
  };
};

function collectRequestedLimits(adapter: AdapterWithLimits, options: CurveGPUContextOptions): CurveGPURequestedLimits {
  const requestedLimits: CurveGPURequestedLimits = {};
  if (options.requireAdapterLimits !== false) {
    if (adapter.limits?.maxStorageBufferBindingSize !== undefined) {
      requestedLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    }
    if (adapter.limits?.maxBufferSize !== undefined) {
      requestedLimits.maxBufferSize = adapter.limits.maxBufferSize;
    }
  }
  if (options.requiredLimits?.maxStorageBufferBindingSize !== undefined) {
    requestedLimits.maxStorageBufferBindingSize = options.requiredLimits.maxStorageBufferBindingSize;
  }
  if (options.requiredLimits?.maxBufferSize !== undefined) {
    requestedLimits.maxBufferSize = options.requiredLimits.maxBufferSize;
  }
  return requestedLimits;
}

function buildDiagnostics(adapter: AdapterWithLimits, adapterInfo: GPUAdapterInfo | null): CurveGPUAdapterDiagnostics {
  return {
    vendor: adapterInfo?.vendor || undefined,
    architecture: adapterInfo?.architecture || undefined,
    description: adapterInfo?.description || undefined,
    isFallbackAdapter: adapter.isFallbackAdapter,
  };
}

/**
 * Create the shared browser WebGPU context for the library.
 *
 * The context owns adapter and device acquisition. It is intended to be
 * created once and reused across field, group, NTT, and MSM operations.
 */
export async function createCurveGPUContext(options: CurveGPUContextOptions = {}): Promise<CurveGPUContext> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }

  const adapter = (await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference,
  })) as AdapterWithLimits | null;
  if (!adapter) {
    throw new Error("requestAdapter returned null");
  }

  const adapterInfo = await getAdapterInfo(adapter);
  const requestedLimits = collectRequestedLimits(adapter, options);
  const device = await adapter.requestDevice({
    requiredLimits: Object.keys(requestedLimits).length > 0 ? requestedLimits : undefined,
  });

  let closed = false;
  return {
    adapter,
    device,
    adapterInfo,
    diagnostics: buildDiagnostics(adapter, adapterInfo),
    requestedLimits,
    close(): void {
      if (closed) {
        return;
      }
      closed = true;
    },
  };
}
