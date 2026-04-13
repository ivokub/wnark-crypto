const logEl = document.getElementById("log");
const runBtn = document.getElementById("runBtn");
const groupEl = document.getElementById("group");
const minLog2El = document.getElementById("minLog2");
const maxLog2El = document.getElementById("maxLog2");
const itersEl = document.getElementById("iters");

function append(line = "") {
  logEl.textContent += `${line}\n`;
}

async function run() {
  const group = groupEl.value === "g2" ? "g2" : "g1";
  logEl.textContent = `=== Heliax ${group.toUpperCase()} MSM Browser Benchmark ===\n\n`;
  runBtn.disabled = true;
  try {
    append("1. Loading wasm module...");
    const wasm = await import("./heliax_wasm_pkg/heliax_browser_bench.js");
    await wasm.default();
    append("1. Loading wasm module... OK");
    append("");

    const minLog2 = Number(minLog2El.value);
    const maxLog2 = Number(maxLog2El.value);
    const iters = Number(itersEl.value);
    const probeFn = group === "g2" ? wasm.probe_msm_g2_once : wasm.probe_msm_once;
    const benchmarkFn = group === "g2" ? wasm.benchmark_msm_g2 : wasm.benchmark_msm;

    append("2. Probing init...");
    const initProbeMs = await wasm.probe_init();
    append(`2. Probing init... OK (${initProbeMs.toFixed(3)} ms)`);
    append("");

    append(`3. Probing single ${group.toUpperCase()} MSM...`);
    const probeSize = 1 << Math.min(minLog2, 10);
    const msmProbeMs = await probeFn(probeSize);
    append(`3. Probing single ${group.toUpperCase()} MSM... OK (n=${probeSize}, ${msmProbeMs.toFixed(3)} ms)`);
    append("");

    append(`4. Running ${group.toUpperCase()} benchmark...`);
    const report = await benchmarkFn(minLog2, maxLog2, iters);
    append("4. Running benchmark... OK");
    append(`init_ms = ${report.init_ms.toFixed(3)}`);
    append("");
    append("size,init_ms,prep_ms,total_ms,total_with_init_ms");
    for (const row of report.rows) {
      append(
        [
          row.size,
          row.init_ms.toFixed(3),
          row.prep_ms.toFixed(3),
          row.total_ms.toFixed(3),
          row.total_with_init_ms.toFixed(3),
        ].join(","),
      );
    }
    append("");
    append(`PASS: Heliax ${group.toUpperCase()} MSM browser benchmark completed`);
  } catch (err) {
    append("");
    append(`FAIL: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", () => {
  void run();
});
