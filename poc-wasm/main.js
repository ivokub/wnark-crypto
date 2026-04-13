import "./bridge.js";

const CURVE_CONFIG = {
  bn254: {
    fixtureBinPath: "/testdata/fixtures/g1/bn254_bases_jacobian.bin",
    coordinateBytes: 32,
    pointBytes: 96,
  },
  bls12_381: {
    fixtureBinPath: "/testdata/fixtures/g1/bls12_381_bases_jacobian.bin",
    coordinateBytes: 48,
    pointBytes: 144,
  },
};

const implSelect = document.getElementById("impl");
const curveSelect = document.getElementById("curve");
const nttLogInput = document.getElementById("ntt-log");
const nttRunsInput = document.getElementById("ntt-runs");
const msmLogInput = document.getElementById("msm-log");
const msmRunsInput = document.getElementById("msm-runs");
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
  const msmLog = Number.parseInt(msmLogInput.value, 10);
  const msmRuns = Number.parseInt(msmRunsInput.value, 10);
  const params = new URLSearchParams(window.location.search);
  const fixtureRev = params.get("fixture-rev") ?? params.get("fixtureRev") ?? "1";
  return {
    curve,
    nttLog,
    nttRuns,
    msmLog,
    msmRuns,
    fixtureBinPath: `${CURVE_CONFIG[curve].fixtureBinPath}?v=${encodeURIComponent(fixtureRev)}`,
    coordinateBytes: CURVE_CONFIG[curve].coordinateBytes,
    pointBytes: CURVE_CONFIG[curve].pointBytes,
  };
}

function applyQueryDefaults() {
  const params = new URLSearchParams(window.location.search);
  const impl = params.get("impl");
  const curve = params.get("curve");
  const nttLog = params.get("ntt-log") ?? params.get("nttLog");
  const nttRuns = params.get("ntt-runs") ?? params.get("nttRuns");
  const msmLog = params.get("msm-log") ?? params.get("msmLog");
  const msmRuns = params.get("msm-runs") ?? params.get("msmRuns");
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
  if (msmLog) {
    msmLogInput.value = msmLog;
  }
  if (msmRuns) {
    msmRunsInput.value = msmRuns;
  }
}

function formatMs(value) {
  return Number(value).toFixed(3);
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
  appendLog("");
  appendLog("--- comparison ---");
  appendLog(`curve: ${a.curve}`);
  appendLog(`ntt size: ${a.ntt_size}`);
  appendLog(`ntt runs: ${a.ntt_runs}`);
  appendLog(`msm size: ${a.msm_size}`);
  appendLog(`msm runs: ${a.msm_runs}`);
  appendLog(`ntt digest match: ${a.ntt_digest_hex === b.ntt_digest_hex}`);
  appendLog(`msm digest match: ${a.msm_digest_hex === b.msm_digest_hex}`);
  appendLog(
    `ntt total ms: webgpu=${formatMs(a.ntt_duration_ms)} pure-go=${formatMs(b.ntt_duration_ms)}`,
  );
  appendLog(
    `ntt avg ms: webgpu=${formatMs(a.ntt_duration_ms / a.ntt_runs)} pure-go=${formatMs(b.ntt_duration_ms / b.ntt_runs)}`,
  );
  appendLog(
    `msm total ms: webgpu=${formatMs(a.msm_duration_ms)} pure-go=${formatMs(b.msm_duration_ms)}`,
  );
  appendLog(
    `msm avg ms: webgpu=${formatMs(a.msm_duration_ms / a.msm_runs)} pure-go=${formatMs(b.msm_duration_ms / b.msm_runs)}`,
  );
}

async function runSelected() {
  clearLog();
  runButton.disabled = true;
  const impl = implSelect.value;
  const config = readConfig();
  const nttSize = 1 << config.nttLog;
  const msmSize = 1 << config.msmLog;

  appendLog("=== CurveGPU Go WASM POC ===");
  appendLog(`impl = ${impl}`);
  appendLog(`curve = ${config.curve}`);
  appendLog(`ntt_size = ${nttSize}`);
  appendLog(`ntt_runs = ${config.nttRuns}`);
  appendLog(`msm_size = ${msmSize}`);
  appendLog(`msm_runs = ${config.msmRuns}`);
  appendLog(`fixture = ${config.fixtureBinPath}`);
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
