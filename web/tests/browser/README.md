# Browser Harness

This directory contains the browser-only smoke and benchmark harnesses for the TypeScript library.

The supported launcher page is:

- `web/tests/browser/curvegpu.html`

For scripted local runs, use:

- `web/tests/browser/run-suite.sh`

Examples:

```sh
make browser-suite CURVE=bn254 SUITE=g1_ops
make browser-suite CURVE=bls12_381 SUITE=fr_ntt
make browser-suite CURVE=bn254 SUITE=g1_msm_bench MIN_LOG=10 MAX_LOG=15 ITERS=1
make browser-suite CURVE=bls12_381 SUITE=g1_msm_bench MIN_LOG=10 MAX_LOG=19 ITERS=1
make browser-suite CURVE=bls12_381 SUITE=g1_msm_bench BASE_SOURCE=generated MIN_LOG=10 MAX_LOG=20 ITERS=1
```

Notes:

- The runner uses a clean temporary Chrome profile for each run.
- By default it runs headless and starts a local `python3 -m http.server` if one is not already serving the repo root.
- `BASE_SOURCE=generated` is useful for MSM benchmark ranges larger than the checked-in fixture.
