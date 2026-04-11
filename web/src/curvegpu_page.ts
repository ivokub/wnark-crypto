export {};

type SuiteKind = "smoke" | "bench";

type SuiteConfig = {
  curve: string;
  id: string;
  label: string;
  kind: SuiteKind;
  title: string;
  description: string;
  script: string;
  defaultMinLog?: number;
  defaultMaxLog?: number;
  defaultIters?: number;
};

const SUITES: SuiteConfig[] = [
  {
    curve: "bn254",
    id: "fr_ops",
    label: "fr ops smoke",
    kind: "smoke",
    title: "BN254 fr WebGPU Smoke",
    description: "Runs the BN254 scalar-field smoke suite against shared vectors.",
    script: "/web/dist/bn254_fr_ops.js",
  },
  {
    curve: "bn254",
    id: "fp_ops",
    label: "fp ops smoke",
    kind: "smoke",
    title: "BN254 fp WebGPU Smoke",
    description: "Runs the BN254 base-field smoke suite against shared vectors.",
    script: "/web/dist/bn254_fp_ops.js",
  },
  {
    curve: "bn254",
    id: "fr_vector_ops",
    label: "fr vector smoke",
    kind: "smoke",
    title: "BN254 fr Vector WebGPU Smoke",
    description: "Runs the BN254 vector-operation smoke suite.",
    script: "/web/dist/bn254_fr_vector_ops.js",
  },
  {
    curve: "bn254",
    id: "fr_ntt",
    label: "fr NTT smoke",
    kind: "smoke",
    title: "BN254 fr NTT Browser Smoke",
    description: "Runs the BN254 NTT smoke suite.",
    script: "/web/dist/bn254_fr_ntt.js",
  },
  {
    curve: "bn254",
    id: "g1_ops",
    label: "G1 ops smoke",
    kind: "smoke",
    title: "BN254 G1 Phase 6 Browser Smoke",
    description: "Runs the BN254 G1 point-operation smoke suite.",
    script: "/web/dist/bn254_g1_ops.js",
  },
  {
    curve: "bn254",
    id: "g1_scalar_mul",
    label: "G1 scalar mul smoke",
    kind: "smoke",
    title: "BN254 G1 Phase 7 Browser Smoke",
    description: "Runs the BN254 G1 scalar-multiplication smoke suite.",
    script: "/web/dist/bn254_g1_scalar_mul.js",
  },
  {
    curve: "bn254",
    id: "g1_msm",
    label: "G1 MSM smoke",
    kind: "smoke",
    title: "BN254 G1 Phase 8 Browser Smoke",
    description: "Runs the BN254 G1 MSM smoke suite.",
    script: "/web/dist/bn254_g1_msm.js",
  },
  {
    curve: "bn254",
    id: "fr_vector_bench",
    label: "fr vector benchmark",
    kind: "bench",
    title: "BN254 fr Vector WebGPU Benchmark",
    description: "Measures BN254 vector operations end-to-end in browser WebGPU.",
    script: "/web/dist/bn254_fr_vector_bench.js",
    defaultMinLog: 10,
    defaultMaxLog: 20,
    defaultIters: 3,
  },
  {
    curve: "bn254",
    id: "fr_ntt_bench",
    label: "fr NTT benchmark",
    kind: "bench",
    title: "BN254 fr NTT Browser Benchmark",
    description: "Measures BN254 NTT end-to-end in browser WebGPU.",
    script: "/web/dist/bn254_fr_ntt_bench.js",
    defaultMinLog: 10,
    defaultMaxLog: 14,
    defaultIters: 1,
  },
  {
    curve: "bn254",
    id: "g1_msm_bench",
    label: "G1 MSM benchmark",
    kind: "bench",
    title: "BN254 G1 MSM Browser Benchmark",
    description: "Measures BN254 MSM end-to-end in browser WebGPU.",
    script: "/web/dist/bn254_g1_msm_bench.js",
    defaultMinLog: 10,
    defaultMaxLog: 20,
    defaultIters: 1,
  },
  {
    curve: "bls12_381",
    id: "fr_ops",
    label: "fr ops smoke",
    kind: "smoke",
    title: "BLS12-381 fr Phase 2 Browser Smoke",
    description: "Runs the BLS12-381 scalar-field smoke suite against shared vectors.",
    script: "/web/dist/bls12_381_fr_ops.js",
  },
  {
    curve: "bls12_381",
    id: "fp_ops",
    label: "fp ops smoke",
    kind: "smoke",
    title: "BLS12-381 fp Phase 3 Browser Smoke",
    description: "Runs the BLS12-381 base-field smoke suite against shared vectors.",
    script: "/web/dist/bls12_381_fp_ops.js",
  },
  {
    curve: "bls12_381",
    id: "g1_ops",
    label: "G1 ops smoke",
    kind: "smoke",
    title: "BLS12-381 G1 Phase 6 Browser Smoke",
    description: "Runs the BLS12-381 G1 point-operation smoke suite.",
    script: "/web/dist/bls12_381_g1_ops.js",
  },
  {
    curve: "bls12_381",
    id: "g1_scalar_mul",
    label: "G1 scalar mul smoke",
    kind: "smoke",
    title: "BLS12-381 G1 Phase 7 Browser Smoke",
    description: "Runs the BLS12-381 G1 scalar-multiplication smoke suite.",
    script: "/web/dist/bls12_381_g1_scalar_mul.js",
  },
  {
    curve: "bls12_381",
    id: "g1_msm",
    label: "G1 MSM smoke",
    kind: "smoke",
    title: "BLS12-381 G1 Phase 8 Browser Smoke",
    description: "Runs the BLS12-381 G1 MSM smoke suite.",
    script: "/web/dist/bls12_381_g1_msm.js",
  },
  {
    curve: "bls12_381",
    id: "g1_msm_bench",
    label: "G1 MSM benchmark",
    kind: "bench",
    title: "BLS12-381 G1 MSM Browser Benchmark",
    description: "Measures BLS12-381 MSM end-to-end in browser WebGPU.",
    script: "/web/dist/bls12_381_g1_msm_bench.js",
    defaultMinLog: 10,
    defaultMaxLog: 19,
    defaultIters: 1,
  },
];

function getById<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLElement)) {
    throw new Error(`missing element: ${id}`);
  }
  return el as T;
}

function suiteKey(curve: string, suiteId: string): string {
  return `${curve}:${suiteId}`;
}

function suiteMap(): Map<string, SuiteConfig> {
  const map = new Map<string, SuiteConfig>();
  for (const suite of SUITES) {
    map.set(suiteKey(suite.curve, suite.id), suite);
  }
  return map;
}

function readRequestedSuite(): SuiteConfig {
  const params = new URLSearchParams(window.location.search);
  const explicitSuite = params.get("suite");
  const curve = params.get("curve") ?? "bn254";
  const map = suiteMap();
  if (explicitSuite) {
    const exact = map.get(suiteKey(curve, explicitSuite));
    if (exact) {
      return exact;
    }
    const legacyDirect = SUITES.find((suite) => `${suite.curve}_${suite.id}` === explicitSuite);
    if (legacyDirect) {
      return legacyDirect;
    }
  }
  return map.get(suiteKey("bn254", "fr_ops")) ?? SUITES[0];
}

function populateSelectors(curveSelect: HTMLSelectElement, suiteSelect: HTMLSelectElement, selected: SuiteConfig): void {
  const curves = [...new Set(SUITES.map((suite) => suite.curve))];
  curveSelect.replaceChildren();
  for (const curve of curves) {
    const option = document.createElement("option");
    option.value = curve;
    option.textContent = curve;
    if (curve === selected.curve) {
      option.selected = true;
    }
    curveSelect.appendChild(option);
  }

  const selectedCurveSuites = SUITES.filter((suite) => suite.curve === selected.curve);
  suiteSelect.replaceChildren();
  for (const suite of selectedCurveSuites) {
    const option = document.createElement("option");
    option.value = suite.id;
    option.textContent = `${suite.label}`;
    if (suite.id === selected.id) {
      option.selected = true;
    }
    suiteSelect.appendChild(option);
  }
}

function updateSuiteOptions(suiteSelect: HTMLSelectElement, curve: string, selectedSuiteId: string | null): void {
  suiteSelect.replaceChildren();
  for (const suite of SUITES.filter((entry) => entry.curve === curve)) {
    const option = document.createElement("option");
    option.value = suite.id;
    option.textContent = suite.label;
    if (suite.id === selectedSuiteId || (!selectedSuiteId && suiteSelect.options.length === 0)) {
      option.selected = true;
    }
    suiteSelect.appendChild(option);
  }
}

function applyPageConfig(selected: SuiteConfig): void {
  document.title = selected.title;
  getById<HTMLElement>("heading").textContent = selected.title;
  getById<HTMLElement>("description").textContent = selected.description;
  getById<HTMLElement>("suite-kind").textContent = selected.kind === "bench" ? "Benchmark" : "Smoke";

  const benchControls = getById<HTMLElement>("bench-controls");
  if (selected.kind === "bench") {
    benchControls.hidden = false;
    const minLogEl = getById<HTMLInputElement>("min-log");
    const maxLogEl = getById<HTMLInputElement>("max-log");
    const itersEl = getById<HTMLInputElement>("iters");
    if (selected.defaultMinLog !== undefined) {
      minLogEl.value = `${selected.defaultMinLog}`;
    }
    if (selected.defaultMaxLog !== undefined) {
      maxLogEl.value = `${selected.defaultMaxLog}`;
    }
    if (selected.defaultIters !== undefined) {
      itersEl.value = `${selected.defaultIters}`;
    }
  } else {
    benchControls.hidden = true;
  }
}

function wireSuiteSelector(selected: SuiteConfig): void {
  const curveSelect = getById<HTMLSelectElement>("curve-select");
  const suiteSelect = getById<HTMLSelectElement>("suite-select");
  const openButton = getById<HTMLButtonElement>("open-suite");
  populateSelectors(curveSelect, suiteSelect, selected);

  curveSelect.addEventListener("change", () => {
    updateSuiteOptions(suiteSelect, curveSelect.value, null);
  });

  openButton.addEventListener("click", () => {
    const params = new URLSearchParams(window.location.search);
    params.set("curve", curveSelect.value);
    params.set("suite", suiteSelect.value);
    window.location.search = params.toString();
  });
}

async function main(): Promise<void> {
  const selected = readRequestedSuite();
  applyPageConfig(selected);
  wireSuiteSelector(selected);
  await import(`${selected.script}?v=1`);
}

void main();
