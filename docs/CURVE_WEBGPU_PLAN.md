# Pairing-Friendly Curve WebGPU Plan

Status: BN254 baseline complete; BN254 browser performance tracks benchmarked; BLS12-381 `fr`, `fp`, G1, scalar-mul, and MSM baselines complete; optimized browser MSM paths exist for BN254 and BLS12-381; Heliax browser/native BLS12-381 MSM wrapper benchmarked
Last updated: 2026-04-12

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
- Curve manifests and codegen-backed layout scaffolding are started for multi-curve support.
- Shared browser MSM layers now exist for:
  - metadata construction
  - GPU runtime helpers
  - benchmark helpers
  - benchmark page runner
  - sparse signed Pippenger benchmark driver
- Shared browser benchmark base-source helpers now exist for both BN254 and BLS12-381.
- A unified browser suite launcher now exists at `web/static/curvegpu.html`.
- Shared browser suite entrypoints now exist for:
  - `fr` ops
  - `fp` ops
  - G1 ops
  - G1 scalar multiplication
  - G1 MSM smoke
  - G1 MSM benchmark
- Browser MSM benchmark base selection now follows one shared priority order:
  - fixture if available
  - otherwise benchmark server if available
  - otherwise generated bases
- A shared `pkg/testgen` package now exists for reusable benchmark/test data generation.
- `go generate ./testdata` now regenerates the tracked curve testdata from repo root.

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
- A Go benchmark server now exists and serves deterministic BN254 and BLS12-381 base sets for browser MSM runs.

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

### 1. Finish the shared multi-curve browser MSM framework

Continue reducing BN254/BLS12-381 duplication so the two paths differ mostly by curve config:

- point packing / unpacking
- shader paths and entrypoint sets
- vector or fixture sources
- field/point buffer sizes
- window-selection policy if curve-specific

The target shape is a config-driven or generated MSM driver rather than handwritten per-curve benchmark pages.

### 2. Expand the Go-backed benchmark data server and fixture model

The benchmark server is in place and already serves BN254 and BLS12-381 base sets, but the overall data-source model still needs refinement. The static `2^19` BLS12-381 fixture remains useful as a proving-key-style benchmark, but it has limits:

- fixed maximum size
- large static artifact
- awkward regeneration flow

Next benchmark infrastructure task:

- keep growing the server-backed path as the primary large-run source
- continue generating valid random or deterministic base points in Go via `gnark-crypto`
- support larger prefix slices on demand for browser benchmarks
- keep the existing static fixture path as a fallback

This should better model the practical workflow of loading proving-key-like base sets without forcing us to check in larger and larger blobs.

### 3. Move more generator logic into `pkg/testgen`

The first generator extraction is in place, but most curve testdata still lives inside separate `cmd/*-vector-gen` implementations.

Next generator work:

- move more vector/fixture construction into `pkg/testgen`
- keep the command binaries as thin wrappers around shared generators
- reuse the same generators from the benchmark server where practical
- keep `go generate ./testdata` as the single regeneration entrypoint

This is the direction needed to match the `gnark-crypto/internal/generator` style more closely.

### 4. Push the optimized BLS12-381 MSM path into shared/runtime code

The BLS12-381 optimized browser path is correct and fast enough to treat as a real backend now. The next engineering step is to make it a first-class shared pipeline rather than a benchmark-only special case.

### 5. Extend multi-curve generation beyond shape/layout scaffolding

Adopt more of the `gnark-crypto/internal/generator` style:

- shared curve manifests
- generated constants and layout bindings
- generated curve-specific wrapper/config layers
- minimize hand-maintained per-curve boilerplate

### 6. Clean up placeholder and debugging files

There are still temporary or debugging artifacts in the repo that should be removed or consolidated once the current refactor settles:

- stale one-off static harness copies
- temporary benchmark/debug assets no longer used
- placeholder files kept only for early bring-up
- old per-suite page wrappers that are superseded by the unified launcher

This cleanup should happen after the current shared-driver refactor is stable, so we do not delete files still serving as temporary references.

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
- Metal native smokes are used where stable
- broad repeated native GPU-heavy test sweeps are treated cautiously when `gogpu/wgpu` backend instability shows up

## Immediate Next Step

Continue extracting the optimized browser MSM pipeline into a curve-config-driven shared driver, while moving more curve testdata generation into `pkg/testgen` and removing obsolete one-off browser harness files as they become superseded.
