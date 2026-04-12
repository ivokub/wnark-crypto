# Pairing-Friendly Curve WebGPU Plan

Status: browser multi-curve library scaffold complete; public TS API in place; browser smoke/bench harnesses moved onto the library API; BN254 and BLS12-381 `fr`, `fp`, G1, NTT, scalar-mul, and MSM browser paths validated; optimized browser MSM paths exist for BN254 and BLS12-381; Heliax browser/native BLS12-381 MSM wrapper benchmarked
Last updated: 2026-04-13

## Goal

Build a reusable WebGPU primitive layer for pairing-friendly curves, shared across:

- Go + `gogpu/wgpu` on Metal where stable
- TypeScript + browser WebGPU

The current target curves are:

- BN254
- BLS12-381
- BLS12-377 later

This remains a primitive-layer project. It is not yet a Groth16 prover or pairing implementation.

## Completed Work

### Shared infrastructure

- Shared 4-word and 6-word host/GPU field layouts are in place.
- Shared curve layout metadata is in place for BN254, BLS12-381, and BLS12-377 shape support.
- Shared browser MSM layers now exist for:
  - metadata construction
  - GPU runtime helpers
  - benchmark helpers
  - benchmark page runner
  - sparse signed Pippenger benchmark driver
- Shared browser benchmark base-source helpers now exist for both BN254 and BLS12-381.
- A unified browser suite launcher now exists at `web/tests/browser/curvegpu.html`.
- A scripted browser harness runner now exists at `web/tests/browser/run-suite.sh` with Make targets for repeatable local validation.
- Shared browser suite entrypoints now exist for:
  - `fr` ops
  - `fp` ops
  - G1 ops
  - G1 scalar multiplication
  - G1 MSM smoke
  - G1 MSM benchmark
- A public browser library entrypoint now exists at `web/src/index.ts`.
- Public browser WebGPU API layers now exist for:
  - shared context creation
  - curve module creation
  - `fr` / `fp` field modules
  - G1 operations
  - scalar-field NTT
  - G1 MSM
- Browser harness pages now consume the library API rather than owning shader/runtime logic directly.
- A consumer-oriented browser example now exists at `web/examples/library_example.html`.
- Browser MSM benchmark base selection now follows one shared priority order:
  - fixture if available
  - otherwise generated bases
- A shared `pkg/testgen` package now exists for reusable benchmark/test data generation.
- `go generate ./testdata` now regenerates the tracked curve testdata through a single repo entrypoint: `cmd/curvegpu-testdata-gen`.
- `cmd/curvegpu-testdata-gen` now calls `pkg/testgen` directly for field ops, vector ops, G1 ops, scalar-mul vectors, MSM vectors, BN254 NTT data, and G1 base fixtures.
- The old per-dataset `cmd/*-vector-gen` and `cmd/*-ntt-domain-gen` wrappers have been removed.

### BN254

- `fr` baseline: complete and browser/Metal validated
- `fp` baseline: complete and browser/Metal validated
- `fr` vector ops baseline: complete and browser validated
- `fr` NTT baseline: complete and browser validated
- G1 baseline arithmetic: complete and browser/Metal validated
- scalar multiplication baseline: complete and browser validated
- MSM baseline: complete and browser validated
- optimized browser MSM path:
  - sparse signed bucketing
  - chunking
  - parallel merge/reduction stages
  - shared benchmark/runtime refactors applied

### BLS12-381

- `fr` baseline: complete and browser/Metal validated
- `fp` baseline: complete and browser/Metal validated
- `fr` vector ops baseline: complete and browser validated
- `fr` NTT baseline: complete and browser validated
- G1 baseline arithmetic: complete and browser/Metal validated
- scalar multiplication baseline: complete and browser validated
- MSM baseline: complete and browser validated
- optimized browser MSM path:
  - fixed base fixture loading
  - sparse signed Pippenger pipeline
  - weighted-bucket plus split subsum window reduction
  - correctness-validated against the naive/reference MSM smoke path

### External reference work

- Local Heliax `webgpu-groth16` MSM wrappers were benchmarked natively and in browser.
- This gave an apples-to-apples BLS12-381 browser comparison against our framework.

## Current Benchmark Picture

### BN254 browser MSM

The optimized BN254 browser MSM path is materially faster than the naive baseline at large sizes and is already in a reasonable place for browser-only use.

### BLS12-381 browser MSM vs Heliax

Validated browser numbers on the same machine currently show:

- Heliax is ahead through roughly `2^16`
- our optimized path overtakes around `2^17`
- our optimized path is materially ahead at `2^18` and `2^19`

Representative BLS12-381 browser MSM timings:

- our path:
  - `2^16`: `725.9 ms`
  - `2^17`: `705.7 ms`
  - `2^18`: `942.5 ms`
  - `2^19`: `1474.2 ms`
- Heliax wrapper:
  - `2^16`: `438.3 ms`
  - `2^17`: `809.9 ms`
  - `2^18`: `1625.1 ms`
  - `2^19`: `3271.9 ms`

This means Heliax remains an important architectural reference, but the current browser-only results do not justify replacing our implementation outright.

## Near-Term Priorities

### 1. Polish the production browser library surface

The library boundary now exists, but it still needs more productization work:

- tighten a few remaining public API rough edges
- decide what low-level primitives should remain internal vs public
- add more consumer-facing documentation and examples
- eventually package the browser entrypoint as a more explicit distributable module

The goal is for benchmarks and smokes to remain harnesses, while the maintained entrypoint for users is the typed library API.

### 2. Keep refining the shared multi-curve browser runtime

Most browser runtime duplication is gone, but there is still room to simplify the remaining curve-config seams so the implementation differs mostly by declarative curve metadata.

### 3. Expand the local fixture-generation model

The checked-in BLS12-381 fixture remains useful as a proving-key-style benchmark, but the overall local fixture model still needs refinement:

- fixed maximum size
- large static artifact
- awkward regeneration flow

Next benchmark infrastructure task:

- keep using valid curve points for correctness-preserving benchmarks
- use checked-in fixtures where they exist
- let users generate larger local fixtures with `go generate ./testdata` or `make fixture-...`
- keep generated in-browser bases only as a convenience fallback where no fixture is present at all

This should better model the practical workflow of loading proving-key-like base sets without requiring an auxiliary server.

### 4. Consolidate the remaining generation model around `pkg/testgen`

The user-facing generator surface is now where it should be:

- `go generate ./testdata`
- or directly `go run ./cmd/curvegpu-testdata-gen`

Remaining work here is narrower:

- move toward a more manifest-driven generation model
- adopt more of the `gnark-crypto/internal/generator` style for constants and curve-specific bindings

### 5. Push the optimized BLS12-381 MSM path into shared/runtime code

The BLS12-381 optimized browser path is correct and fast enough to treat as a real backend now. The next engineering step is to make it a first-class shared pipeline rather than a benchmark-only special case.

### 6. Extend multi-curve generation beyond shape/layout scaffolding

Adopt more of the `gnark-crypto/internal/generator` style:

- shared curve manifests
- generated constants and layout bindings
- generated curve-specific wrapper/config layers
- minimize hand-maintained per-curve boilerplate

### 7. Continue repository cleanup as refactors settle

The large browser-page cleanup is already done, but there is still normal follow-up cleanup to keep doing as the framework stabilizes:

- remove generated/build artifacts from version control
- keep one supported browser launcher path
- delete temporary bring-up helpers once the shared runtime replaces them
- avoid reintroducing per-curve wrappers when a shared entrypoint is already sufficient

## Later Optimization Tracks

These are explicitly later than the current shared-framework work:

- GLV-based scalar decomposition where curve support makes sense
- more Heliax-style bucketing/subsum refinements
- better native Go/Metal benchmarking once `gogpu/wgpu` instability is less constraining
- 13-bit-limb arithmetic experiments, if we decide to trade representation simplicity for throughput
- BLS12-377 bring-up using the same 4-word `fr` / 6-word `fp` framework

## Validation Policy

Current practical validation policy remains:

- browser WebGPU on headed Chrome is the primary correctness and performance signal
- a clean-profile Chrome automation path now exists for repeatable local smoke and benchmark runs
- Metal native smokes are used where stable
- native BLS12-381 `fr` vector/NTT bench commands now exist, but repeated native GPU-heavy runs still hit the same `gogpu/wgpu` Metal `resource already released` instability, so they are not yet treated as authoritative performance signals
- broad repeated native GPU-heavy test sweeps are treated cautiously when `gogpu/wgpu` backend instability shows up

## Immediate Next Step

Stabilize the library-facing browser API as the primary maintained entrypoint, while continuing to shrink the remaining curve-specific runtime seams underneath it.
