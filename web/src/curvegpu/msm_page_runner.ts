type BenchmarkElements = {
  minLogEl: HTMLInputElement;
  maxLogEl: HTMLInputElement;
  itersEl: HTMLInputElement;
  runButton: HTMLButtonElement;
};

type BenchmarkUi = {
  setStatus: (text: string) => void;
  setPageState: (state: string) => void;
  writeLog: (lines: string[]) => void;
};

type BenchmarkInitResult<T> = {
  context: T;
  preMetricLines?: string[];
  postMetricLines?: string[];
};

export async function runMSMBenchmarkPage<T>(options: {
  title: string;
  successMessage: string;
  tableHeader: string;
  elements: BenchmarkElements;
  ui: BenchmarkUi;
  appendAdapterDiagnostics: (adapter: GPUAdapter, lines: string[]) => Promise<void>;
  init: (args: { adapter: GPUAdapter; device: GPUDevice; lines: string[] }) => Promise<BenchmarkInitResult<T>>;
  runSizes: (args: {
    context: T;
    lines: string[];
    initMs: number;
    minLog: number;
    maxLog: number;
    iters: number;
    writeLog: (lines: string[]) => void;
  }) => Promise<void>;
}): Promise<void> {
  const { elements, ui } = options;
  const lines = [`=== ${options.title} ===`, ""];
  ui.writeLog(lines);
  ui.setStatus("Running");
  ui.setPageState("running");
  elements.runButton.disabled = true;

  try {
    const minLog = Number.parseInt(elements.minLogEl.value, 10);
    const maxLog = Number.parseInt(elements.maxLogEl.value, 10);
    const iters = Number.parseInt(elements.itersEl.value, 10);
    if (!Number.isInteger(minLog) || !Number.isInteger(maxLog) || !Number.isInteger(iters) || minLog < 1 || maxLog < minLog || iters < 1) {
      throw new Error("invalid benchmark controls");
    }
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }

    const initStart = performance.now();
    lines.push("1. Requesting adapter... OK");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    await options.appendAdapterDiagnostics(adapter, lines);
    const adapterWithLimits = adapter as GPUAdapter & {
      limits?: {
        maxStorageBufferBindingSize?: number;
        maxBufferSize?: number;
      };
    };
    const requiredLimits: Record<string, number> = {};
    if (adapterWithLimits.limits?.maxStorageBufferBindingSize !== undefined) {
      requiredLimits.maxStorageBufferBindingSize = adapterWithLimits.limits.maxStorageBufferBindingSize;
    }
    if (adapterWithLimits.limits?.maxBufferSize !== undefined) {
      requiredLimits.maxBufferSize = adapterWithLimits.limits.maxBufferSize;
    }

    lines.push("2. Requesting device... OK");
    const device = await adapter.requestDevice({
      requiredLimits: Object.keys(requiredLimits).length > 0 ? requiredLimits : undefined,
    });

    const init = await options.init({ adapter, device, lines });
    const initMs = performance.now() - initStart;
    if (init.preMetricLines) {
      lines.push(...init.preMetricLines);
    }
    lines.push(`init_ms = ${initMs.toFixed(3)}`);
    if (init.postMetricLines) {
      lines.push(...init.postMetricLines);
    }
    lines.push("");
    lines.push(options.tableHeader);

    await options.runSizes({
      context: init.context,
      lines,
      initMs,
      minLog,
      maxLog,
      iters,
      writeLog: ui.writeLog,
    });

    lines.push("");
    lines.push(`PASS: ${options.successMessage}`);
    ui.writeLog(lines);
    ui.setStatus("Pass");
    ui.setPageState("pass");
  } catch (error) {
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    ui.writeLog(lines);
    ui.setStatus("Fail");
    ui.setPageState("fail");
  } finally {
    elements.runButton.disabled = false;
  }
}
