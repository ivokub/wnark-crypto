const FP_BYTES = 32;
const SCALAR_BYTES = 32;
const POINT_BYTES = 96;
const UNIFORM_BYTES = 32;
const ZERO_HEX = "0000000000000000000000000000000000000000000000000000000000000000";
const G1_OP_DOUBLE_JAC = 4;
const G1_OP_ADD_MIXED = 5;
const G1_OP_JAC_TO_AFFINE = 6;
const G1_OP_AFFINE_ADD = 7;
const G1_OP_SCALAR_MUL_AFFINE = 8;

const runButton = document.getElementById("run");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

function setStatus(text) { statusEl.textContent = text; }
function setPageState(state) { document.body.dataset.status = state; }
function writeLog(lines) { logEl.textContent = lines.join("\n"); }

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function fetchVectors() {
  const text = await fetchText("/testdata/vectors/g1/bn254_phase8_msm.json");
  return JSON.parse(text);
}

async function getAdapterInfo(adapter) {
  if (adapter.info) {
    return adapter.info;
  }
  if (typeof adapter.requestAdapterInfo === "function") {
    try {
      return await adapter.requestAdapterInfo();
    } catch {
      return null;
    }
  }
  return null;
}

async function appendAdapterDiagnostics(adapter, lines) {
  if ("isFallbackAdapter" in adapter) {
    lines.push(`adapter.isFallbackAdapter = ${String(adapter.isFallbackAdapter)}`);
  }
  const info = await getAdapterInfo(adapter);
  if (!info) {
    lines.push("adapter.info = unavailable");
    return;
  }
  if (info.vendor) {
    lines.push(`adapter.vendor = ${info.vendor}`);
  }
  if (info.architecture) {
    lines.push(`adapter.architecture = ${info.architecture}`);
  }
}

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function zeroPoint() {
  return { x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX };
}

function isInfinityPoint(point) {
  return point.x_bytes_le === ZERO_HEX && point.y_bytes_le === ZERO_HEX;
}

function createKernel(device, shaderCode, entryPoint = "g1_ops_main") {
  const shaderModule = device.createShaderModule({
    label: "bn254-g1-shader",
    code: shaderCode,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    label: "bn254-g1-bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "bn254-g1-pl",
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    label: "bn254-g1-pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint },
  });
  return { pipeline, bindGroupLayout };
}

function affineToKernelPoint(point, oneMontZ) {
  const zero = "0000000000000000000000000000000000000000000000000000000000000000";
  const isInfinity = point.x_bytes_le === zero && point.y_bytes_le === zero;
  return {
    x_bytes_le: point.x_bytes_le,
    y_bytes_le: point.y_bytes_le,
    z_bytes_le: isInfinity ? zero : oneMontZ,
  };
}

function packPointBatch(points) {
  const out = new Uint8Array(points.length * POINT_BYTES);
  points.forEach((point, index) => {
    const base = index * POINT_BYTES;
    out.set(hexToBytes(point.x_bytes_le), base);
    out.set(hexToBytes(point.y_bytes_le), base + FP_BYTES);
    out.set(hexToBytes(point.z_bytes_le), base + 2 * FP_BYTES);
  });
  return out;
}

function scalarToKernelPoint(scalar) {
  return {
    x_bytes_le: scalar,
    y_bytes_le: ZERO_HEX,
    z_bytes_le: ZERO_HEX,
  };
}

function unpackPointBatch(bytes, count) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * POINT_BYTES;
    out.push({
      x_bytes_le: bytesToHex(bytes.slice(base, base + FP_BYTES)),
      y_bytes_le: bytesToHex(bytes.slice(base + FP_BYTES, base + 2 * FP_BYTES)),
      z_bytes_le: bytesToHex(bytes.slice(base + 2 * FP_BYTES, base + 3 * FP_BYTES)),
    });
  }
  return out;
}

function expectPointBatch(name, got, want) {
  if (got.length !== want.length) {
    throw new Error(`${name}: length mismatch got=${got.length} want=${want.length}`);
  }
  for (let i = 0; i < got.length; i += 1) {
    if (got[i].x_bytes_le !== want[i].x_bytes_le ||
      got[i].y_bytes_le !== want[i].y_bytes_le ||
      got[i].z_bytes_le !== want[i].z_bytes_le) {
      throw new Error(
        `${name}: mismatch at index ${i}` +
        ` got=(${got[i].x_bytes_le},${got[i].y_bytes_le},${got[i].z_bytes_le})` +
        ` want=(${want[i].x_bytes_le},${want[i].y_bytes_le},${want[i].z_bytes_le})`
      );
    }
  }
}

function scalarHexLEToBigInt(hex) {
  const bytes = hexToBytes(hex);
  let out = 0n;
  for (let i = bytes.length - 1; i >= 0; i -= 1) {
    out = (out << 8n) | BigInt(bytes[i]);
  }
  return out;
}

function extractWindowDigit(scalar, bitOffset, window) {
  if (window <= 0) {
    return 0;
  }
  const mask = (1n << BigInt(window)) - 1n;
  return Number((scalar >> BigInt(bitOffset)) & mask);
}

function bestPippengerWindow(count) {
  const windows = [4, 5, 6, 7, 8, 9, 10, 11, 12];
  let best = windows[0];
  let bestCost = Number.POSITIVE_INFINITY;
  for (const window of windows) {
    const cost = Math.ceil(255 / window) * (count + (1 << window));
    if (cost < bestCost) {
      bestCost = cost;
      best = window;
    }
  }
  return best;
}

function buildBucketLists(bases, scalars, count, termsPerInstance, window) {
  const numWindows = Math.ceil(256 / window);
  const bucketCount = (1 << window) - 1;
  const bucketLists = Array.from({ length: count * numWindows * bucketCount }, () => []);
  const scalarBigs = scalars.map((scalar) => scalarHexLEToBigInt(scalar));
  for (let instance = 0; instance < count; instance += 1) {
    const baseOffset = instance * termsPerInstance;
    for (let term = 0; term < termsPerInstance; term += 1) {
      const idx = baseOffset + term;
      const scalar = scalarBigs[idx];
      for (let win = 0; win < numWindows; win += 1) {
        const digit = extractWindowDigit(scalar, win * window, window);
        if (digit === 0) {
          continue;
        }
        const bucketIndex = ((instance * numWindows + win) * bucketCount) + (digit - 1);
        bucketLists[bucketIndex].push(bases[idx]);
      }
    }
  }
  return bucketLists;
}

async function runOp(device, kernel, opcode, inputA, inputB, extraParams) {
  const count = (extraParams && extraParams.count) || inputA.length;
  if (count === 0) {
    return [];
  }
  const aBytes = packPointBatch(inputA);
  const bBytes = packPointBatch(inputB.length === 0 ? [zeroPoint()] : inputB);
  const inputCount = Math.max(inputA.length, inputB.length === 0 ? 1 : inputB.length, count);
  const inputByteSize = inputCount * POINT_BYTES;
  const outputByteSize = count * POINT_BYTES;

  const basesBuffer = device.createBuffer({ label: "g1-input-a", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const scalarsBuffer = device.createBuffer({ label: "g1-input-b", size: inputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ label: "g1-output", size: outputByteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ label: "g1-staging", size: outputByteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniformBuffer = device.createBuffer({ label: "g1-params", size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  device.queue.writeBuffer(basesBuffer, 0, aBytes);
  device.queue.writeBuffer(scalarsBuffer, 0, bBytes);
  const params = new Uint32Array(UNIFORM_BYTES / 4);
  params[0] = count;
  params[1] = opcode;
  params[2] = (extraParams && extraParams.termsPerInstance) || 0;
  params[3] = (extraParams && extraParams.window) || 0;
  params[4] = (extraParams && extraParams.numWindows) || 0;
  params[5] = (extraParams && extraParams.bucketCount) || 0;
  params[6] = (extraParams && extraParams.rowWidth) || 0;
  device.queue.writeBuffer(uniformBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    label: "g1-msm-bind-group",
    layout: kernel.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: basesBuffer } },
      { binding: 1, resource: { buffer: scalarsBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: "g1-msm-encoder" });
  const pass = encoder.beginComputePass({ label: "g1-msm-pass" });
  pass.setPipeline(kernel.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputByteSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  basesBuffer.destroy();
  scalarsBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();
  uniformBuffer.destroy();

  return unpackPointBatch(result, count);
}

async function runMSM(device, kernel, bases, scalars, count, termsPerInstance) {
  let state = await runOp(device, kernel, G1_OP_SCALAR_MUL_AFFINE, bases, scalars.map((scalar) => scalarToKernelPoint(scalar)));
  let width = termsPerInstance;
  while (width > 1) {
    const nextWidth = Math.ceil(width / 2);
    const left = [];
    const right = [];
    for (let instance = 0; instance < count; instance += 1) {
      const rowBase = instance * width;
      for (let j = 0; j < nextWidth; j += 1) {
        left.push(state[rowBase + 2 * j]);
        right.push(2 * j + 1 < width ? state[rowBase + 2 * j + 1] : { x_bytes_le: ZERO_HEX, y_bytes_le: ZERO_HEX, z_bytes_le: ZERO_HEX });
      }
    }
    state = await runOp(device, kernel, G1_OP_AFFINE_ADD, left, right);
    width = nextWidth;
  }
  return state;
}

async function reduceAffineBuckets(device, kernel, bucketLists) {
  let current = bucketLists.map((bucket) => bucket.slice());
  for (;;) {
    const next = Array.from({ length: current.length }, () => []);
    const left = [];
    const right = [];
    const mappings = [];
    let work = false;
    for (let bucketIndex = 0; bucketIndex < current.length; bucketIndex += 1) {
      const bucket = current[bucketIndex];
      if (bucket.length <= 1) {
        next[bucketIndex] = bucket.slice();
        continue;
      }
      work = true;
      const nextCount = Math.ceil(bucket.length / 2);
      next[bucketIndex] = new Array(nextCount);
      for (let j = 0; j < nextCount; j += 1) {
        left.push(bucket[2 * j]);
        right.push(2 * j + 1 < bucket.length ? bucket[2 * j + 1] : zeroPoint());
        mappings.push({ bucket: bucketIndex, slot: j });
      }
    }
    if (!work) {
      return current.map((bucket) => (bucket.length === 1 ? bucket[0] : zeroPoint()));
    }
    const reduced = await runOp(device, kernel, G1_OP_AFFINE_ADD, left, right);
    for (let i = 0; i < reduced.length; i += 1) {
      const mapping = mappings[i];
      next[mapping.bucket][mapping.slot] = reduced[i];
    }
    current = next;
  }
}

async function reduceWindows(device, kernel, bucketSums, count, numWindows, bucketCount) {
  const totalSlots = count * numWindows;
  const running = Array.from({ length: totalSlots }, () => zeroPoint());
  const totals = Array.from({ length: totalSlots }, () => zeroPoint());

  for (let bucket = bucketCount - 1; bucket >= 0; bucket -= 1) {
    const activeRunningIdx = [];
    const activeRunning = [];
    const activeBuckets = [];
    for (let slot = 0; slot < totalSlots; slot += 1) {
      const point = bucketSums[slot * bucketCount + bucket];
      if (isInfinityPoint(point)) {
        continue;
      }
      activeRunningIdx.push(slot);
      activeRunning.push(running[slot]);
      activeBuckets.push(point);
    }
    if (activeRunningIdx.length > 0) {
      const nextRunning = await runOp(device, kernel, G1_OP_AFFINE_ADD, activeRunning, activeBuckets);
      for (let i = 0; i < activeRunningIdx.length; i += 1) {
        running[activeRunningIdx[i]] = nextRunning[i];
      }
    }

    const activeTotalIdx = [];
    const activeTotals = [];
    const activeRunningTotals = [];
    for (let slot = 0; slot < totalSlots; slot += 1) {
      if (isInfinityPoint(running[slot])) {
        continue;
      }
      activeTotalIdx.push(slot);
      activeTotals.push(totals[slot]);
      activeRunningTotals.push(running[slot]);
    }
    if (activeTotalIdx.length > 0) {
      const nextTotals = await runOp(device, kernel, G1_OP_AFFINE_ADD, activeTotals, activeRunningTotals);
      for (let i = 0; i < activeTotalIdx.length; i += 1) {
        totals[activeTotalIdx[i]] = nextTotals[i];
      }
    }
  }

  return totals;
}

async function combineWindows(device, kernel, windowSums, count, numWindows, window) {
  let acc = Array.from({ length: count }, () => zeroPoint());
  const zeros = Array.from({ length: count }, () => zeroPoint());
  for (let win = numWindows - 1; win >= 0; win -= 1) {
    if (win !== numWindows - 1) {
      for (let step = 0; step < window; step += 1) {
        acc = await runOp(device, kernel, G1_OP_DOUBLE_JAC, acc, zeros);
      }
    }
    const activeIdx = [];
    const activeAcc = [];
    const activeAff = [];
    for (let instance = 0; instance < count; instance += 1) {
      const point = windowSums[instance * numWindows + win];
      if (isInfinityPoint(point)) {
        continue;
      }
      activeIdx.push(instance);
      activeAcc.push(acc[instance]);
      activeAff.push(point);
    }
    if (activeIdx.length === 0) {
      continue;
    }
    const nextAcc = await runOp(device, kernel, G1_OP_ADD_MIXED, activeAcc, activeAff);
    for (let i = 0; i < activeIdx.length; i += 1) {
      acc[activeIdx[i]] = nextAcc[i];
    }
  }
  return runOp(device, kernel, G1_OP_JAC_TO_AFFINE, acc, zeros);
}

async function runPippengerMSM(device, kernel, bases, scalars, count, termsPerInstance, window) {
  const bucketCount = (1 << window) - 1;
  const numWindows = Math.ceil(256 / window);
  const bucketSums = await runOp(
    device,
    kernel.bucket,
    0,
    bases,
    scalars.map((scalar) => scalarToKernelPoint(scalar)),
    {
      count: count * numWindows * bucketCount,
      termsPerInstance,
      window,
      numWindows,
      bucketCount,
    },
  );
  let windowSums = await runOp(
    device,
    kernel.weight,
    0,
    bucketSums,
    [],
    {
      count: count * numWindows * bucketCount,
      termsPerInstance,
      window,
      numWindows,
      bucketCount,
    },
  );
  let rowWidth = bucketCount;
  while (rowWidth > 1) {
    const nextWidth = Math.ceil(rowWidth / 2);
    windowSums = await runOp(
      device,
      kernel.reduce,
      0,
      windowSums,
      [],
      {
        count: count * numWindows * nextWidth,
        rowWidth,
      },
    );
    rowWidth = nextWidth;
  }
  return runOp(
    device,
    kernel.combine,
    0,
    windowSums,
    [],
    {
      count,
      termsPerInstance,
      window,
      numWindows,
      bucketCount,
    },
  );
}

async function runPippengerMSMHost(device, kernel, bases, scalars, count, termsPerInstance, window) {
  const numWindows = Math.ceil(256 / window);
  const bucketCount = (1 << window) - 1;
  const bucketLists = buildBucketLists(bases, scalars, count, termsPerInstance, window);
  const bucketSums = await reduceAffineBuckets(device, kernel, bucketLists);
  const windowSums = await reduceWindows(device, kernel, bucketSums, count, numWindows, bucketCount);
  const result = await combineWindows(device, kernel, windowSums, count, numWindows, window);
  return { bucketSums, windowSums, result };
}

async function runSmoke() {
  const lines = ["=== BN254 G1 Phase 8 Browser Smoke ===", ""];
  writeLog(lines);
  setStatus("Running");
  setPageState("running");
  runButton.disabled = true;

  try {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("requestAdapter returned null");
    }
    lines.push("1. Requesting adapter... OK");
    await appendAdapterDiagnostics(adapter, lines);

    const device = await adapter.requestDevice();
    lines.push("2. Requesting device... OK");

    const [shaderText, vectors] = await Promise.all([
      fetchText("/shaders/curves/bn254/g1_arith.wgsl"),
      fetchVectors(),
    ]);
    lines.push("3. Loading shader and vectors... OK");
    lines.push(`terms_per_instance = ${vectors.terms_per_instance}`);
    lines.push(`cases.msm = ${vectors.msm_cases.length}`);

    const kernel = {
      ops: createKernel(device, shaderText),
      bucket: createKernel(device, shaderText, "g1_msm_bucket_main"),
      weight: createKernel(device, shaderText, "g1_msm_window_weight_main"),
      reduce: createKernel(device, shaderText, "g1_msm_window_reduce_main"),
      combine: createKernel(device, shaderText, "g1_msm_combine_main"),
    };
    lines.push("4. Creating pipeline... OK");

    const bases = [];
    const scalars = [];
    const want = [];
    for (const msmCase of vectors.msm_cases) {
      for (const base of msmCase.bases_affine) {
        bases.push(affineToKernelPoint(base, vectors.one_mont_z));
      }
      for (const scalar of msmCase.scalars_bytes_le) {
        scalars.push(scalar);
      }
      want.push(msmCase.expected_affine);
    }

    expectPointBatch("msm_naive_affine", await runMSM(device, kernel.ops, bases, scalars, vectors.msm_cases.length, vectors.terms_per_instance), want);
    lines.push("msm_naive_affine: OK");

    const window = bestPippengerWindow(vectors.terms_per_instance);
    const bucketCount = (1 << window) - 1;
    const numWindows = Math.ceil(256 / window);
    const hostPippenger = await runPippengerMSMHost(
      device,
      kernel.ops,
      bases,
      scalars,
      vectors.msm_cases.length,
      vectors.terms_per_instance,
      window,
    );
    expectPointBatch("msm_pippenger_host", hostPippenger.result, want);
    lines.push(`msm_pippenger_host (window=${window}): OK`);

    const gpuBucketSums = await runOp(
      device,
      kernel.bucket,
      0,
      bases,
      scalars.map((scalar) => scalarToKernelPoint(scalar)),
      {
        count: vectors.msm_cases.length * numWindows * bucketCount,
        termsPerInstance: vectors.terms_per_instance,
        window,
        numWindows,
        bucketCount,
      },
    );
    expectPointBatch("msm_pippenger_bucket_stage", gpuBucketSums, hostPippenger.bucketSums);
    lines.push(`msm_pippenger_bucket_stage (window=${window}): OK`);

    let gpuWindowSums = await runOp(
      device,
      kernel.weight,
      0,
      gpuBucketSums,
      [],
      {
        count: vectors.msm_cases.length * numWindows * bucketCount,
        termsPerInstance: vectors.terms_per_instance,
        window,
        numWindows,
        bucketCount,
      },
    );
    let rowWidth = bucketCount;
    while (rowWidth > 1) {
      const nextWidth = Math.ceil(rowWidth / 2);
      gpuWindowSums = await runOp(
        device,
        kernel.reduce,
        0,
        gpuWindowSums,
        [],
        {
          count: vectors.msm_cases.length * numWindows * nextWidth,
          rowWidth,
        },
      );
      rowWidth = nextWidth;
    }
    expectPointBatch("msm_pippenger_window_stage", gpuWindowSums, hostPippenger.windowSums);
    lines.push(`msm_pippenger_window_stage (window=${window}): OK`);

    expectPointBatch(
      "msm_pippenger_affine",
      await runOp(
        device,
        kernel.combine,
        0,
        gpuWindowSums,
        [],
        {
          count: vectors.msm_cases.length,
          termsPerInstance: vectors.terms_per_instance,
          window,
          numWindows,
          bucketCount,
        },
      ),
      want,
    );
    lines.push(`msm_pippenger_affine (window=${window}): OK`);

    lines.push("");
    lines.push("PASS: BN254 G1 Phase 8 browser smoke succeeded");
    writeLog(lines);
    setStatus("PASS");
    setPageState("pass");
  } catch (error) {
    lines.push("");
    lines.push(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    writeLog(lines);
    setStatus("FAIL");
    setPageState("fail");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void runSmoke();
});
