# POC WASM

This directory contains a narrow proof-of-concept for the intended browser
architecture:

- `Go wasm -> syscall/js -> JS bridge -> TS WebGPU library`
- compared against
- `Go wasm -> gnark-crypto`

The scope is intentionally small. It is only meant to validate that:

- Go wasm can drive the browser WebGPU library cleanly
- the same pattern works for both `bn254` and `bls12_381`
- FFT/NTT, G1 MSM, and G2 MSM can be compared against a pure Go wasm baseline

## Build

From repo root:

```sh
make poc-wasm-build
```

That will:

- build the browser TS library into `web/dist/`
- compile the two Go wasm binaries into `poc-wasm/dist/`
- copy `wasm_exec.js` into `poc-wasm/dist/`

## Run

Serve the repo root, for example:

```sh
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/poc-wasm/index.html
```

The page supports query-driven autorun, for example:

```text
http://localhost:8000/poc-wasm/index.html?autorun=1&impl=both&curve=bn254&ntt-log=12&ntt-runs=4&g1-msm-log=12&g1-msm-runs=4&g2-msm-log=11&g2-msm-runs=2
```

Parameters:

- `impl=webgpu-go|gnark-go|both`
- `curve=bn254|bls12_381`
- `ntt-log=<power-of-two log size>`
- `ntt-runs=<repeat count>`
- `g1-msm-log=<power-of-two log size>`
- `g1-msm-runs=<repeat count>`
- `g2-msm-log=<power-of-two log size>`
- `g2-msm-runs=<repeat count>`
- `fixture-rev=<cache bust token>`

Legacy `msm-log` / `msm-runs` query parameters are still accepted and applied to
both G1 and G2 MSM when the new per-group parameters are omitted.

## Larger MSM runs

The checked-in fixtures are intentionally small (`2^15` points). For larger
local runs, overwrite the local fixture with a bigger one:

```sh
make poc-wasm-fixture-bn254 COUNT=131072
make poc-wasm-fixture-bls12_381 COUNT=131072
```

Example sizes:

- `COUNT=65536` for `msm-log=16`
- `COUNT=131072` for `msm-log=17`
- `COUNT=262144` for `msm-log=18`

Convenience targets for `2^18`:

```sh
make poc-wasm-fixture-bn254-2pow18
make poc-wasm-fixture-bls12_381-2pow18
```

After regenerating a fixture, bump `fixture-rev` in the page URL so the browser
does not reuse the older cached file:

```text
http://localhost:8000/poc-wasm/index.html?autorun=1&impl=both&curve=bn254&ntt-log=12&ntt-runs=4&g1-msm-log=17&g1-msm-runs=4&g2-msm-log=15&g2-msm-runs=2&fixture-rev=2
```

Suggested `2^18` runs:

```text
http://localhost:8000/poc-wasm/index.html?autorun=1&impl=both&curve=bn254&ntt-log=18&ntt-runs=4&g1-msm-log=18&g1-msm-runs=4&g2-msm-log=16&g2-msm-runs=2&fixture-rev=3
http://localhost:8000/poc-wasm/index.html?autorun=1&impl=both&curve=bls12_381&ntt-log=18&ntt-runs=4&g1-msm-log=18&g1-msm-runs=4&g2-msm-log=16&g2-msm-runs=2&fixture-rev=3
```

Notes:

- The pure-Go baseline uses local `gnark-crypto` through the repo `replace`.
- This is a POC harness, not part of the library public API.
- The batch mode reuses one WebGPU context and one loaded fixture across all runs.
- The WebGPU path performs one untimed prewarm pass before measured runs, then
  reports `startup`, `steady-state`, and `overall` timings separately.
