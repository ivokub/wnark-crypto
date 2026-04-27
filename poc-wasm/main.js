import "./bridge.js";

const CURVE_CONFIG = {
  bn254: {
    g1FixtureBinPath: "/testdata/fixtures/g1/bn254_bases_jacobian.bin",
    g2FixtureBinPath: "/testdata/fixtures/g2/bn254_bases_jacobian.bin",
  },
  bls12_381: {
    g1FixtureBinPath: "/testdata/fixtures/g1/bls12_381_bases_jacobian.bin",
    g2FixtureBinPath: "/testdata/fixtures/g2/bls12_381_bases_jacobian.bin",
  },
};

const implSelect = document.getElementById("impl");
const curveSelect = document.getElementById("curve");
const nttLogInput = document.getElementById("ntt-log");
const nttRunsInput = document.getElementById("ntt-runs");
const g1MsmLogInput = document.getElementById("g1-msm-log");
const g1MsmRunsInput = document.getElementById("g1-msm-runs");
const g2MsmLogInput = document.getElementById("g2-msm-log");
const g2MsmRunsInput = document.getElementById("g2-msm-runs");
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

function readConfig() {
  const curve = curveSelect.value;
  const nttLog = Number.parseInt(nttLogInput.value, 10);
  const nttRuns = Number.parseInt(nttRunsInput.value, 10);
  const g1MsmLog = Number.parseInt(g1MsmLogInput.value, 10);
  const g1MsmRuns = Number.parseInt(g1MsmRunsInput.value, 10);
  const g2MsmLog = Number.parseInt(g2MsmLogInput.value, 10);
  const g2MsmRuns = Number.parseInt(g2MsmRunsInput.value, 10);
  const params = new URLSearchParams(window.location.search);
  return {
    curve,
    nttLog,
    nttRuns,
    g1MsmLog,
    g1MsmRuns,
    g2MsmLog,
    g2MsmRuns,
    g1FixtureBinPath: `${CURVE_CONFIG[curve].g1FixtureBinPath}`,
    g2FixtureBinPath: `${CURVE_CONFIG[curve].g2FixtureBinPath}`,
  };
}

function applyQueryDefaults() {
  const params = new URLSearchParams(window.location.search);
  const impl = params.get("impl");
  const curve = params.get("curve");
  const nttLog = params.get("ntt-log") ?? params.get("nttLog");
  const nttRuns = params.get("ntt-runs") ?? params.get("nttRuns");
  const legacyMsmLog = params.get("msm-log") ?? params.get("msmLog");
  const legacyMsmRuns = params.get("msm-runs") ?? params.get("msmRuns");
  const g1MsmLog = params.get("g1-msm-log") ?? params.get("g1MsmLog") ?? legacyMsmLog;
  const g1MsmRuns = params.get("g1-msm-runs") ?? params.get("g1MsmRuns") ?? legacyMsmRuns;
  const g2MsmLog = params.get("g2-msm-log") ?? params.get("g2MsmLog") ?? legacyMsmLog;
  const g2MsmRuns = params.get("g2-msm-runs") ?? params.get("g2MsmRuns") ?? legacyMsmRuns;
  if (impl && ["webgpu-go", "gnark-go", "both"].includes(impl)) {
    implSelect.value = impl;
  }
  if (curve && ["bn254", "bls12_381"].includes(curve)) {
    curveSelect.value = curve;
  }
  if (nttLog) {
    nttLogInput.value = nttLog;
  }
  if (nttRuns) {
    nttRunsInput.value = nttRuns;
  }
  if (g1MsmLog) {
    g1MsmLogInput.value = g1MsmLog;
  }
  if (g1MsmRuns) {
    g1MsmRunsInput.value = g1MsmRuns;
  }
  if (g2MsmLog) {
    g2MsmLogInput.value = g2MsmLog;
  }
  if (g2MsmRuns) {
    g2MsmRunsInput.value = g2MsmRuns;
  }
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

async function ensureGoRuntime() {
  if (typeof window.Go === "function") {
    return;
  }
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/poc-wasm/dist/wasm_exec.js";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("failed to load wasm_exec.js"));
    document.head.appendChild(script);
  });
}

async function runGoWasm(impl, wasmPath, config) {
  await ensureGoRuntime();
  appendLog(`--- ${impl} ---`);
  const go = new window.Go();
  const resultPromise = new Promise((resolve, reject) => {
    window.__curvegpuPocConfig = config;
    window.__curvegpuPocLog = (line) => appendLog(String(line));
    window.__curvegpuPocSetStatus = (text) => setStatus(String(text));
    window.__curvegpuPocComplete = (result) => resolve(result);
    window.__curvegpuPocFail = (message) => reject(new Error(String(message)));
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

function compareResults(a, b) {
  const webgpuStartupMs = readMs(a, "startup_duration_ms", readMs(a, "init_duration_ms"));
  const pureGoStartupMs = readMs(b, "startup_duration_ms", readMs(b, "fixture_load_ms"));
  const webgpuSteadyStateMs = readMs(
    a,
    "steady_state_duration_ms",
    readMs(a, "ntt_duration_ms") + readMs(a, "g1_msm_duration_ms") + readMs(a, "g2_msm_duration_ms"),
  );
  const pureGoSteadyStateMs = readMs(
    b,
    "steady_state_duration_ms",
    readMs(b, "ntt_duration_ms") + readMs(b, "g1_msm_duration_ms") + readMs(b, "g2_msm_duration_ms"),
  );
  const webgpuOverallMs = readMs(a, "overall_duration_ms", readMs(a, "total_duration_ms"));
  const pureGoOverallMs = readMs(b, "overall_duration_ms", readMs(b, "total_duration_ms"));

  appendLog("");
  appendLog("--- comparison ---");
  appendLog(`curve: ${a.curve}`);
  appendLog(`ntt size: ${a.ntt_size}`);
  appendLog(`ntt runs: ${a.ntt_runs}`);
  appendLog(`g1 msm size: ${a.g1_msm_size}`);
  appendLog(`g1 msm runs: ${a.g1_msm_runs}`);
  appendLog(`g2 msm size: ${a.g2_msm_size}`);
  appendLog(`g2 msm runs: ${a.g2_msm_runs}`);
  appendLog(`ntt digest match: ${a.ntt_digest_hex === b.ntt_digest_hex}`);
  appendLog(`g1 msm digest match: ${a.g1_msm_digest_hex === b.g1_msm_digest_hex}`);
  appendLog(`g2 msm digest match: ${a.g2_msm_digest_hex === b.g2_msm_digest_hex}`);
  appendLog(
    `startup ms: webgpu=${formatMs(webgpuStartupMs)} pure-go=${formatMs(pureGoStartupMs)}`,
  );
  if (typeof a.prewarm_duration_ms === "number") {
    appendLog(
      `webgpu startup breakdown: init=${formatMs(readMs(a, "init_duration_ms"))} prewarm=${formatMs(readMs(a, "prewarm_duration_ms"))}`,
    );
  }
  if (typeof b.g1_fixture_load_ms === "number" || typeof b.g2_fixture_load_ms === "number") {
    appendLog(
      `pure-go startup breakdown: g1_fixture=${formatMs(readMs(b, "g1_fixture_load_ms"))} g2_fixture=${formatMs(readMs(b, "g2_fixture_load_ms"))}`,
    );
  }
  appendLog(
    `steady-state total ms: webgpu=${formatMs(webgpuSteadyStateMs)} pure-go=${formatMs(pureGoSteadyStateMs)}`,
  );
  appendLog(
    `overall total ms: webgpu=${formatMs(webgpuOverallMs)} pure-go=${formatMs(pureGoOverallMs)}`,
  );
  appendLog(
    `ntt total ms: webgpu=${formatMs(a.ntt_duration_ms)} pure-go=${formatMs(b.ntt_duration_ms)}`,
  );
  appendLog(
    `ntt avg ms: webgpu=${formatMs(a.ntt_duration_ms / a.ntt_runs)} pure-go=${formatMs(b.ntt_duration_ms / b.ntt_runs)}`,
  );
  appendLog(
    `g1 msm total ms: webgpu=${formatMs(a.g1_msm_duration_ms)} pure-go=${formatMs(b.g1_msm_duration_ms)}`,
  );
  appendLog(
    `g1 msm avg ms: webgpu=${formatMs(a.g1_msm_duration_ms / a.g1_msm_runs)} pure-go=${formatMs(b.g1_msm_duration_ms / b.g1_msm_runs)}`,
  );
  appendLog(
    `g2 msm total ms: webgpu=${formatMs(a.g2_msm_duration_ms)} pure-go=${formatMs(b.g2_msm_duration_ms)}`,
  );
  appendLog(
    `g2 msm avg ms: webgpu=${formatMs(a.g2_msm_duration_ms / a.g2_msm_runs)} pure-go=${formatMs(b.g2_msm_duration_ms / b.g2_msm_runs)}`,
  );
}

async function runSelected() {
  clearLog();
  runButton.disabled = true;
  const impl = implSelect.value;
  const config = readConfig();
  const nttSize = 1 << config.nttLog;
  const g1MsmSize = 1 << config.g1MsmLog;
  const g2MsmSize = 1 << config.g2MsmLog;

  appendLog("=== CurveGPU Go WASM POC ===");
  appendLog(`impl = ${impl}`);
  appendLog(`curve = ${config.curve}`);
  appendLog(`ntt_size = ${nttSize}`);
  appendLog(`ntt_runs = ${config.nttRuns}`);
  appendLog(`g1_msm_size = ${g1MsmSize}`);
  appendLog(`g1_msm_runs = ${config.g1MsmRuns}`);
  appendLog(`g2_msm_size = ${g2MsmSize}`);
  appendLog(`g2_msm_runs = ${config.g2MsmRuns}`);
  appendLog(`g1_fixture = ${config.g1FixtureBinPath}`);
  appendLog(`g2_fixture = ${config.g2FixtureBinPath}`);
  appendLog("note = WebGPU timings are reported after an untimed prewarm run; compare steady-state lines for throughput.");
  appendLog("");
  setStatus("Running");

  try {
    let webgpuResult = null;
    let gnarkResult = null;

    if (impl === "webgpu-go" || impl === "both") {
      webgpuResult = await runGoWasm("webgpu-go", "/poc-wasm/dist/go-webgpu.wasm", config);
    }
    if (impl === "gnark-go" || impl === "both") {
      gnarkResult = await runGoWasm("gnark-go", "/poc-wasm/dist/go-gnark.wasm", config);
    }
    if (webgpuResult && gnarkResult) {
      compareResults(webgpuResult, gnarkResult);
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

const params = new URLSearchParams(window.location.search);
if (params.get("autorun") === "1") {
  void runSelected();
}
