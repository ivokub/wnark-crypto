import "/backend/accelerated/webgpu/groth16/bridge.js";

const implSelect = document.getElementById("impl");
const curveSelect = document.getElementById("curve");
const sizeLogSelect = document.getElementById("size-log");
const proveRunsInput = document.getElementById("prove-runs");
const runButton = document.getElementById("run");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

function appendLog(line = "") {
  logEl.textContent += `${line}\n`;
}

function clearLog() {
  logEl.textContent = "";
}

function setStatus(text) {
  statusEl.textContent = text;
}

function formatMs(value) {
  return Number(value).toFixed(3);
}

function readMs(result, key, fallback = 0) {
  if (result && typeof result[key] === "number") {
    return result[key];
  }
  return fallback;
}

function readConfig() {
  return {
    curve: curveSelect.value,
    sizeLog: Number.parseInt(sizeLogSelect.value, 10),
    proveRuns: Number.parseInt(proveRunsInput.value, 10),
  };
}

function applyQueryDefaults() {
  const params = new URLSearchParams(window.location.search);
  const impl = params.get("impl");
  const curve = params.get("curve");
  const sizeLog = params.get("size-log") ?? params.get("sizeLog");
  const proveRuns = params.get("prove-runs") ?? params.get("proveRuns");

  if (impl && ["both", "webgpu-go", "native-go"].includes(impl)) {
    implSelect.value = impl;
  }
  if (curve && ["bn254", "bls12_377", "bls12_381"].includes(curve)) {
    curveSelect.value = curve;
  }
  if (sizeLog && ["12", "15", "18"].includes(sizeLog)) {
    sizeLogSelect.value = sizeLog;
  }
  if (proveRuns) {
    proveRunsInput.value = proveRuns;
  }
}

async function ensureGoRuntime() {
  if (typeof window.Go === "function") {
    return;
  }
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/poc-gnark-groth16/dist/wasm_exec.js";
    script.onload = resolve;
    script.onerror = () => reject(new Error("failed to load wasm_exec.js"));
    document.head.appendChild(script);
  });
}

async function runGoWasm(impl, wasmPath, config) {
  await ensureGoRuntime();
  appendLog(`--- ${impl} ---`);

  const go = new window.Go();
  const resultPromise = new Promise((resolve, reject) => {
    window.__wnarkGroth16PocConfig = config;
    window.__wnarkGroth16PocLog = (line) => appendLog(String(line));
    window.__wnarkGroth16PocSetStatus = (text) => setStatus(String(text));
    window.__wnarkGroth16PocComplete = (result) => resolve(result);
    window.__wnarkGroth16PocFail = (message) => reject(new Error(String(message)));
  });

  const response = await fetch(wasmPath);
  if (!response.ok) {
    throw new Error(`failed to fetch ${wasmPath}: ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, go.importObject);
  const runResult = go.run(instance);
  const result = await resultPromise;
  if (runResult && typeof runResult.then === "function") {
    await runResult;
  }
  return result;
}

function compareResults(webgpu, nativeImpl) {
  appendLog("");
  appendLog("--- comparison ---");
  appendLog(`curve: ${webgpu.curve}`);
  appendLog(`fixture: 2^${webgpu.size_log}`);
  appendLog(`depth: ${webgpu.depth_size}`);
  appendLog(`prove runs: ${webgpu.prove_runs}`);
  appendLog(`constraints: ${webgpu.constraints}`);
  appendLog(
    `roundtrip verify: webgpu=${webgpu.roundtrip_verify_succeeded} native=${nativeImpl.roundtrip_verify_succeeded}`,
  );
  appendLog(
    `proof size bytes: webgpu=${webgpu.proof_size_bytes} native=${nativeImpl.proof_size_bytes}`,
  );
  appendLog(
    `startup ms: webgpu=${formatMs(webgpu.startup_duration_ms)} native=${formatMs(nativeImpl.startup_duration_ms)}`,
  );
  appendLog(
    `startup breakdown: webgpu fixture=${formatMs(readMs(webgpu, "fixture_duration_ms"))} witness=${formatMs(readMs(webgpu, "witness_duration_ms"))} prepare=${formatMs(readMs(webgpu, "prepare_duration_ms"))} | native fixture=${formatMs(readMs(nativeImpl, "fixture_duration_ms"))} witness=${formatMs(readMs(nativeImpl, "witness_duration_ms"))}`,
  );
  appendLog(
    `steady-state total ms: webgpu=${formatMs(webgpu.steady_state_duration_ms)} native=${formatMs(nativeImpl.steady_state_duration_ms)}`,
  );
  appendLog(
    `overall total ms: webgpu=${formatMs(webgpu.overall_duration_ms)} native=${formatMs(nativeImpl.overall_duration_ms)}`,
  );
  appendLog(
    `prove avg ms: webgpu=${formatMs(webgpu.prove_duration_ms / webgpu.prove_runs)} native=${formatMs(nativeImpl.prove_duration_ms / nativeImpl.prove_runs)}`,
  );
  appendLog(
    `serialize avg ms: webgpu=${formatMs(webgpu.serialize_duration_ms / webgpu.prove_runs)} native=${formatMs(nativeImpl.serialize_duration_ms / nativeImpl.prove_runs)}`,
  );
  appendLog(
    `verify avg ms: webgpu=${formatMs(webgpu.verify_duration_ms / webgpu.prove_runs)} native=${formatMs(nativeImpl.verify_duration_ms / nativeImpl.prove_runs)}`,
  );
}

async function runSelected() {
  clearLog();
  runButton.disabled = true;
  const impl = implSelect.value;
  const config = readConfig();

  appendLog("=== Groth16 Go WASM POC ===");
  appendLog(`impl = ${impl}`);
  appendLog(`curve = ${config.curve}`);
  appendLog(`fixture = 2^${config.sizeLog}`);
  appendLog(`prove_runs = ${config.proveRuns}`);
  appendLog("note = Proof bytes are not compared because prover randomness is expected. Each path loads a fixed serialized circuit and keys, then validates by WriteTo -> ReadFrom -> Verify.");
  appendLog("");

  setStatus("Running");
  try {
    let webgpuResult = null;
    let nativeResult = null;

    if (impl === "webgpu-go" || impl === "both") {
      webgpuResult = await runGoWasm("webgpu-go", "/poc-gnark-groth16/dist/go-webgpu.wasm", config);
    }
    if (impl === "native-go" || impl === "both") {
      nativeResult = await runGoWasm("native-go", "/poc-gnark-groth16/dist/go-native.wasm", config);
    }
    if (webgpuResult && nativeResult) {
      compareResults(webgpuResult, nativeResult);
    }
    setStatus("PASS");
  } catch (error) {
    setStatus("FAIL");
    appendLog("");
    appendLog(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", () => {
  void runSelected();
});

applyQueryDefaults();

if (new URLSearchParams(window.location.search).get("autorun") === "1") {
  void runSelected();
}
