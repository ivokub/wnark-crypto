export {};

type PageConfig = {
  title: string;
  heading: string;
  description: string;
  script: string;
};

const BENCH_PAGES: Record<string, PageConfig> = {
  bn254: {
    title: "BN254 G1 MSM Browser Benchmark",
    heading: "BN254 G1 MSM Browser Benchmark",
    description:
      "Unified MSM benchmark page. Use query params to select the curve and benchmark base source.",
    script: "/web/dist/bn254_g1_msm_bench.js",
  },
  bls12_381: {
    title: "BLS12-381 G1 MSM Browser Benchmark",
    heading: "BLS12-381 G1 MSM Browser Benchmark",
    description:
      "Unified MSM benchmark page. Use query params to select the curve and benchmark base source.",
    script: "/web/dist/bls12_381_g1_msm_bench.js",
  },
};

function getCurve(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("curve") ?? "bn254";
}

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`missing element: ${id}`);
  }
  el.textContent = text;
}

async function main(): Promise<void> {
  const curve = getCurve();
  const config = BENCH_PAGES[curve];
  if (!config) {
    throw new Error(`unsupported curve: ${curve}`);
  }
  document.title = config.title;
  setText("heading", config.heading);
  setText("description", config.description);
  await import(`${config.script}?v=1`);
}

void main();
