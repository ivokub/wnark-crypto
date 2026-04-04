# Pairing-Friendly Curve WebGPU Primitive Plan

Status: Phase 0, Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, and Phase 8 completed; Performance Tracks 1, 2, and 3 benchmarked; Heliax browser/native MSM wrapper benchmarked; Phase 9 started with codegen-backed multi-curve scaffolding
Last updated: 2026-04-04

## Goal

Build a correctness-first WebGPU arithmetic prototype for pairing-friendly curves, with BN254 as the first implementation target, covering the primitives needed for later zk-SNARK work:

- scalar-field arithmetic over `fr`
- base-field arithmetic over `fp`
- elliptic-curve arithmetic for G1
- FFT / NTT over `fr`
- later, batched operations such as MSM

This plan explicitly does **not** implement Groth16 itself yet. The objective is to make the primitive layer correct, testable, and reusable from both:

- Go + `gogpu/wgpu` on Metal
- TypeScript + browser WebGPU

## Non-goals for the first iterations

- no full prover integration
- no pairing implementation
- no G2 implementation in the first phase
- no aggressive GPU-specific optimization before correctness is established
- no attempt at constant-time or side-channel guarantees on GPU in the prototype phase

## Curve support target

The architecture should support more than BN254. BN254 remains the first curve to implement, but the interfaces and file layout should be designed so that BLS12-381 and BLS12-377 can be added without redesign.

Curves we should plan for:

- BN254
- BLS12-381
- BLS12-377

Observed gnark-crypto field sizes:

- BN254:
  - `fr` = `[4]uint64`
  - `fp` = `[4]uint64`
- BLS12-381:
  - `fr` = `[4]uint64`
  - `fp` = `[6]uint64`
- BLS12-377:
  - `fr` = `[4]uint64`
  - `fp` = `[6]uint64`

Implications:

- the scalar-field path can likely share a common 4-word backend across all three curves
- the base-field path must support both 4-word and 6-word host representations
- host wrappers, shader constants, and test harnesses must be curve-parameterized from the start

## Current direction

The next implementation step is an apples-to-apples multi-curve port, starting with BLS12-381 in this repo's framework.

The first slice of that work is not arithmetic yet. It is:

- one generated curve manifest shared by Go, TypeScript, and WGSL type layouts
- BLS12-381 representation and conversion coverage in Go against `gnark-crypto`
- keeping BN254 and BLS12-377 on the same generated shape path

This mirrors the config-driven/code-generation style used by `gnark-crypto/internal/generator` and is intended to avoid hand-porting shape metadata curve by curve.

## Source references

Primary reference implementations and call sites:

- [`prove.go`](/Users/ivo/Work/consensys/repo/zk/gnark/backend/groth16/bn254/prove.go)
- [`fr/element.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fr/element.go)
- [`fr/element_purego.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fr/element_purego.go)
- [`fp/element.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fp/element.go)
- [`fp/element_purego.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fp/element_purego.go)
- [`g1.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/g1.go)
- [`fr/fft/domain.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fr/fft/domain.go)
- [`fr/fft/kernel_purego.go`](/Users/ivo/Work/consensys/repo/zk/gnark-crypto/ecc/bn254/fr/fft/kernel_purego.go)

## Key design decisions

### 1. Initial scope order

We should implement in this order:

1. shared infrastructure and curve-parameterized representation
2. BN254 `fr` field element type and basic arithmetic
3. BN254 `fp` field element type and basic arithmetic
4. BN254 `fr` FFT / NTT
5. BN254 G1 point types and point addition / doubling
6. BN254 scalar multiplication
7. BN254 MSM
8. extend reusable pieces to BLS12-381 / BLS12-377

Reason:

- 4-word `fr` is enough to validate the core field machinery and directly supports FFT.
- BN254 `fp` reuses almost the same arithmetic skeleton with different modulus constants.
- G1 depends on `fp`, and MSM depends on stable point arithmetic.
- after the abstractions settle, BLS curves can reuse the same framework with a 6-word base-field backend.

### 2. Representation inside WGSL

WGSL does not provide a practical native `u64` programming model for this work, so the prototype should use a pure `u32` limb representation.

Recommended canonical GPU representation:

- `fr` and `fp` element in Montgomery form
- little-endian limbs
- parameterized `u32` limbs by field size:
  - 4 host words -> `8 x u32`
  - 6 host words -> `12 x u32`

That means:

- BN254:
  - Go `fr.Element` / `fp.Element` = `[4]uint64`
  - GPU `Fr` / `Fp` = `[8]u32`
- BLS12-381 / BLS12-377:
  - Go `fr.Element` = `[4]uint64`
  - Go `fp.Element` = `[6]uint64`
  - GPU `Fr` = `[8]u32`
  - GPU `Fp` = `[12]u32`
- conversion wrappers split each `uint64` into two `u32` limbs

Reason:

- it maps naturally from the existing gnark-crypto representation
- it avoids expensive host-side format churn
- it is straightforward to serialize for Go and TypeScript
- it preserves compatibility with the existing Montgomery arithmetic approach
- it avoids baking BN254’s 4-word base field into the entire design

### 3. Correctness-first arithmetic strategy

For the first implementation pass:

- prefer obvious carry-chain code over clever radix tricks
- keep field elements fully reduced after each public operation
- keep kernels simple and narrowly scoped
- isolate operations into testable helpers before introducing wide batched kernels

This is slower than a tuned GPU design, but it minimizes debugging complexity while we validate semantics.

### 4. Montgomery form

We should stay in Montgomery form internally, matching gnark-crypto.

Needed host-visible conversions:

- canonical bytes / big integer -> Montgomery
- Montgomery -> canonical bytes / regular representation
- Go `fr.Element` / `fp.Element` <-> WGSL words
- TypeScript `Uint32Array` / bytes <-> WGSL words

This keeps the GPU arithmetic compatible with the gnark-crypto internals we are checking against.

### 5. Host API split

We should maintain a shared operation model with two host frontends:

- Go host wrapper for Metal via `gogpu/wgpu`
- TypeScript host wrapper for browser WebGPU

The shaders should be identical between both environments. Only the host binding and buffer management code should differ.

## Proposed repo layout

This is the target structure for the primitive project.

```text
docs/
  BN254_WEBGPU_PLAN.md

shaders/
  common/
    u32_arith.wgsl
    montgomery_u32.wgsl
    montgomery_u32x8.wgsl
    montgomery_u32x12.wgsl
    buffer_layouts.wgsl
  curves/
    bn254/
      fr_types.wgsl
      fr_consts.wgsl
      fr_arith.wgsl
      fp_types.wgsl
      fp_consts.wgsl
      fp_arith.wgsl
      g1_types.wgsl
      g1_arith.wgsl
      fr_ntt.wgsl
    bls12_381/
      fr_types.wgsl
      fr_consts.wgsl
      fp_types.wgsl
      fp_consts.wgsl
    bls12_377/
      fr_types.wgsl
      fr_consts.wgsl
      fp_types.wgsl
      fp_consts.wgsl

go/
  curvegpu/
    shaders.go
    types.go
    buffers.go
    convert.go
    kernels.go
    bn254/
      fr.go
      fp.go
      g1.go
      ntt.go
    bls12_381/
    bls12_377/

web/
  package.json
  tsconfig.json
  src/
    curvegpu/
      convert.ts
      kernels.ts
      bn254/
        types.ts
        fr.ts
        fp.ts
        g1.ts
        ntt.ts
      bls12_381/
      bls12_377/
    example_bn254.ts
  static/
    example_bn254.html

testdata/
  vectors/
    fr/
    fp/
    g1/
    ntt/

cmd/
  bn254-metal-smoke/
  bn254-vector-gen/

tests/
  browser/
  metal/
```

Note:

- the exact path names can be adjusted during implementation
- the main requirement is that WGSL is stored in dedicated files and shared across Go and browser

## Milestones

## Phase 0: Planning and scaffolding

- [x] Create repo plan document
- [x] Confirm plan and initial scope with user
- [x] Create directory layout for shaders, Go host code, TypeScript host code, and testdata
- [x] Make the scaffolding curve-parameterized even if only BN254 is implemented first
- [x] Add shared shader loading strategy for Go and TypeScript
- [x] Add browser and Metal test harness skeletons for future kernels

Deliverable:

- repo structure ready for primitive work, but no arithmetic yet

## Phase 1: Shared representation and conversion layer

- [x] Define WGSL field element structs for 4-word and 6-word fields
- [x] Define Go wrapper types for WGSL buffer-compatible field elements
- [x] Define TypeScript wrapper types for WGSL buffer-compatible field elements
- [x] Implement Go conversion: `[4]uint64` <-> `[8]u32`
- [x] Implement Go conversion: `[6]uint64` <-> `[12]u32`
- [x] Implement TS conversion: bytes / bigint / limb arrays <-> GPU limbs
- [x] Define canonical serialization policy for tests
- [x] Add golden test vectors for edge cases:
  - zero
  - one
  - modulus minus one
  - random values
  - carry-heavy values

Deliverable:

- host can move BN254 field elements between gnark-crypto and WGSL-compatible buffers without ambiguity
- the shared representation layer is ready for future BLS12-381 / BLS12-377 support

Notes from implementation:

- Phase 1 includes shared little-endian test vectors for both 4-word and 6-word layouts.
- Go verification is in place via `go test ./go/curvegpu/...` and the repo-local smoke command in `cmd/curvegpu-metal-smoke`.
- TypeScript scaffolding and conversion helpers are implemented, but no TS build step has been wired up yet because the repo does not currently vendor a local TypeScript toolchain.

## Phase 2: `fr` arithmetic baseline

First field to implement: `fr`.

Operations:

- [x] zero / one / copy / equality
- [x] addition
- [x] subtraction
- [x] negation
- [x] conditional subtraction / normalization
- [x] doubling
- [x] Montgomery multiplication
- [x] squaring
- [x] conversion to Montgomery
- [x] conversion from Montgomery

Kernel strategy:

- [ ] single-op kernels for vectorized elementwise testing
- [ ] tiny smoke kernels for deterministic examples

Tests:

- [x] Go CPU vs GPU random differential tests against gnark-crypto `fr`
- [x] Browser CPU vs GPU random differential tests against exported test vectors
- [x] Metal/browser parity tests on the same vectors
- [x] edge-case tests around carries and reduction boundaries

Deliverable:

- stable `fr` arithmetic layer verified in Metal and browser

Notes from implementation:

- A correctness-first BN254 `fr` WGSL kernel exists for `copy`, `zero`, `one`, `equal`, `add`, `sub`, `neg`, `double`, and `normalize`.
- Metal verification is in place via `CGO_ENABLED=0 go run ./cmd/bn254-fr-metal-smoke`, using shared vectors and direct `gnark-crypto` cross-checks for the reduced-element ops.
- BN254 `fr` Montgomery multiplication is implemented in WGSL using 16-bit sub-limbs inside the shader and is verified on Metal against shared vectors plus `gnark-crypto`.
- BN254 `fr` squaring is implemented by reusing the same Montgomery multiplication path and is verified on Metal against shared vectors plus `gnark-crypto`.
- BN254 `fr` conversion to and from Montgomery form is implemented in WGSL and verified on Metal against shared vectors plus `gnark-crypto`.
- The shared Phase 2 vectors were regenerated from `gnark-crypto` so the browser and Metal harnesses now use the correct BN254 Montgomery encoding for `one`; `equal` returns Montgomery `1` for consistency with the rest of the field API.
- A Go differential test now batches deterministic random BN254 `fr` cases through the WebGPU kernel and checks `copy`, `equal`, `zero`, `one`, `add`, `sub`, `neg`, `double`, `mul`, `square`, `to_mont`, and `from_mont` against `gnark-crypto`.
- Shared browser assets exist (`web/src/bn254_fr_ops.ts`, `web/static/bn254_fr_ops.html`).
- Headed Chrome on Apple Silicon validates the Phase 2 browser smoke successfully across all currently implemented ops, with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- Browser validation now includes `mul: OK`, `square: OK`, `to_mont: OK`, and `from_mont: OK` in the headed-Chrome Phase 2 smoke page, across shared sanity, edge-case, and deterministic differential vectors.
- After adding new browser ops, manual headed-Chrome validation is still required before treating them as browser-verified.
- Headless Chrome on macOS/Metal is currently unreliable for this page and should be treated as a browser automation limitation, not as a correctness failure of the WGSL arithmetic.
- For now, browser validation for Phase 2 is manual via `web/static/bn254_fr_ops.html`; Metal remains the automated local verification path.

## Phase 3: `fp` arithmetic baseline

Implement the same operation set as `fr`, but over `fp`.

- [x] replicate WGSL field logic with `fp` constants
- [x] host wrappers for `fp`
- [x] differential tests against gnark-crypto `fp`
- [x] shared reducer and carry helpers where possible

Deliverable:

- stable `fp` arithmetic layer verified in Metal and browser

Notes from implementation:

- A correctness-first BN254 `fp` WGSL kernel exists for `copy`, `zero`, `one`, `equal`, `add`, `sub`, `neg`, `double`, `mul`, `square`, `to_mont`, `from_mont`, and `normalize`.
- Metal verification is in place via `CGO_ENABLED=0 go run ./cmd/bn254-fp-metal-smoke`, using shared vectors and direct `gnark-crypto` cross-checks for the reduced-element ops.
- A Go differential test batches deterministic random BN254 `fp` cases through the WebGPU kernel and checks `copy`, `equal`, `zero`, `one`, `add`, `sub`, `neg`, `double`, `mul`, `square`, `to_mont`, and `from_mont` against `gnark-crypto`.
- Shared BN254 `fp` vectors now cover sanity cases, carry and reduction edge cases, deterministic differential cases, normalize cases, and Montgomery conversion cases.
- Shared browser assets exist (`web/src/bn254_fp_ops.ts`, `web/static/bn254_fp_ops.html`).
- Headed Chrome on Apple Silicon validates the Phase 3 browser smoke successfully across all currently implemented ops, with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- After adding new browser ops, manual headed-Chrome validation is still required before treating them as browser-verified.
- Headless Chrome on macOS/Metal should still be treated as a browser automation limitation rather than a correctness signal for the WGSL arithmetic pages.
- For now, browser validation for Phase 3 is manual via `web/static/bn254_fp_ops.html`; Metal remains the automated local verification path.

## Phase 4: `fr` vector operations and basic kernels

Once scalar operations are stable:

- [x] vector add / sub / mul kernels
- [x] batched Montgomery conversions
- [x] batched scalar multiply by constants
- [x] permutation / copy / bit-reversal helpers relevant to FFT

Deliverable:

- enough array-oriented `fr` support to support FFT experiments

Notes from implementation:

- The existing BN254 `fr` arithmetic kernel now exposes named batched wrapper methods for `copy`, `add`, `sub`, `mul`, `to_mont`, and `from_mont`, so the scalar field layer is directly usable for vector-sized batches from Go.
- A dedicated BN254 `fr` vector WGSL kernel exists in `shaders/curves/bn254/fr_vector.wgsl` with an initial Phase 4 opcode set covering per-element scaling by factors and bit-reversal copy.
- A Go wrapper for the vector kernel exists in `go/curvegpu/bn254/fr_vector.go`, with `MulFactors` and `BitReverseCopy` entry points.
- A deterministic Go differential test validates the Phase 4 vector slice against CPU expectations in `go/curvegpu/bn254_fr_vector_test.go`.
- A native Metal smoke command exists in `cmd/bn254-fr-vector-metal-smoke`, and it passes on Apple Silicon across `copy`, `add`, `sub`, `mul`, `to_mont`, `from_mont`, `mul_factors`, and `bit_reverse_copy`.
- Shared Phase 4 browser vectors are generated from Go in `testdata/vectors/fr/bn254_phase4_vector_ops.json`.
- A manual headed-Chrome browser smoke exists in `web/static/bn254_fr_vector_ops.html`, and it passes on Apple Silicon with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- As in earlier phases, headed Chrome is the accepted browser validation path for these WebGPU pages; headless Chrome on macOS/Metal is still treated as an automation limitation rather than a correctness signal.

## Phase 5: FFT / NTT over `fr`

Initial goal:

- correctness-first radix-2 power-of-two NTT, matching gnark-crypto domain logic

Tasks:

- [x] mirror domain generation and twiddle handling from gnark-crypto
- [ ] choose first implementation shape:
  - simple iterative DIT/DIF
  - [x] explicit stage-by-stage kernels
- [x] implement bit-reversal strategy
- [x] implement forward NTT
- [x] implement inverse NTT
- [x] verify on multiple domain sizes
- [x] compare against gnark-crypto FFT outputs

Testing:

- [x] small exact cases
- [x] random vectors
- [x] inverse cases
- [ ] coset cases if needed later

Deliverable:

- browser and Metal NTT compatible with gnark-crypto for supported domain sizes

Notes from implementation:

- The implemented Phase 5 slice is a correctness-first BN254 `fr` DIT NTT for small power-of-two domains, built as explicit stage-by-stage kernels rather than a fused FFT pipeline.
- The NTT stage kernel lives in `shaders/curves/bn254/fr_ntt.wgsl`, and the Go wrapper lives in `go/curvegpu/bn254/ntt.go`.
- The current supported validation sizes are `n=8` and `n=16`.
- Shared Phase 5 vectors are generated from `gnark-crypto` in `testdata/vectors/fr/bn254_phase5_ntt.json`, including forward expectations, inverse expectations, per-stage forward twiddle tables, inverse-stage twiddle tables, and the inverse scaling factor.
- A native Metal smoke command exists in `cmd/bn254-fr-ntt-metal-smoke`, and it passes on Apple Silicon for both forward and inverse NTT on the current supported sizes.
- A manual headed-Chrome browser smoke exists in `web/static/bn254_fr_ntt.html`, and it passes on Apple Silicon with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- The current supported validation sizes are still `n=8` and `n=16`, and both forward and inverse paths match `gnark-crypto` on Metal and in headed Chrome.
- The inverse path reuses the vector kernel for the final `n^-1` scaling step instead of baking that scale into the stage shader.
- The Go vector differential test now isolates GPU-heavy subtests onto fresh devices to avoid Metal backend instability during long shared test lifetimes.

## Phase 6: G1 point representation and basic arithmetic

Initial curve scope should be G1 only.

Representations to support:

- [x] affine
- [x] Jacobian

Operations:

- [x] affine infinity encoding policy
- [x] affine/jacobian conversion
- [x] point equality in host-side validation
- [x] affine add
- [x] mixed add
- [x] Jacobian double
- [x] negation
- [x] on-curve checks in host-side tests

Testing:

- [x] compare with gnark-crypto G1 outputs
- [x] include infinity and same-point cases
- [x] include `P + (-P)` and doubling corner cases

Deliverable:

- correctness-first G1 arithmetic baseline

Notes from implementation:

- The first Phase 6 slice implements BN254 G1 affine and Jacobian point layouts in Go and WGSL, with Jacobian-valued GPU outputs used as the canonical point transport format.
- The G1 arithmetic kernel lives in `shaders/curves/bn254/g1_arith.wgsl`, and the Go wrapper lives in `go/curvegpu/bn254/g1.go`.
- The current GPU opcode set covers `copy`, `jac_infinity`, `affine_to_jac`, `neg_jac`, `jac_to_affine`, `double_jac`, `add_mixed`, and `affine_add`.
- Shared Phase 6 vectors are generated from `gnark-crypto` in `testdata/vectors/g1/bn254_phase6_g1_ops.json`, covering infinity, same-point, opposite-point, and deterministic subgroup-point cases.
- A native Metal smoke command exists in `cmd/bn254-g1-metal-smoke`, and it passes on Apple Silicon for the current G1 baseline.
- A manual headed-Chrome browser smoke exists in `web/static/bn254_g1_ops.html`, and it passes on Apple Silicon with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- The Go differential test for this slice lives in `go/curvegpu/bn254_g1_test.go`; it isolates GPU-heavy subtests onto fresh devices to avoid Metal backend instability during long shared test lifetimes.
- Dedicated point-equality logic is still host-side in this baseline. There is no separate GPU equality opcode yet.

## Phase 7: Scalar multiplication

Start with the simplest correct algorithm first.

- [x] fixed-base scalar multiplication baseline
- [x] variable-base double-and-add baseline
- [x] batched scalar multiplication harness
- [x] compare against gnark-crypto `ScalarMultiplication`

Later optimization candidates:

- [ ] windowed scalar multiplication
- [ ] mixed coordinate scheduling
- [ ] GLV decomposition

Deliverable:

- correct scalar multiplication before optimizing MSM

Notes from implementation:

- The first Phase 7 slice adds a correctness-first BN254 G1 scalar-multiplication baseline to `shaders/curves/bn254/g1_arith.wgsl` as `G1_OP_SCALAR_MUL_AFFINE`.
- The GPU path uses a simple MSB-first double-and-add loop over 256 scalar bits, with affine base inputs and affine-normalized outputs.
- The Go wrapper lives in `go/curvegpu/bn254/g1.go` and exposes `ScalarMulAffine` over batched affine bases and packed `fr` scalar words.
- Shared vectors are generated from `gnark-crypto` in `testdata/vectors/g1/bn254_phase7_scalar_mul.json`, covering variable-base and fixed-base cases including zero, one, small deterministic scalars, infinity input, and `q-1`.
- A native Metal smoke command exists in `cmd/bn254-g1-scalar-metal-smoke`, and it passes on Apple Silicon for both `scalar_mul_affine` and `scalar_mul_base_affine`.
- A manual headed-Chrome browser smoke exists in `web/static/bn254_g1_scalar_mul.html`, and it passes on Apple Silicon with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- The Go differential test for this slice lives in `go/curvegpu/bn254_g1_scalar_mul_test.go` and checks the scalar-multiplication results against the shared vectors.
- The broader `CGO_ENABLED=0 go test ./go/curvegpu/...` path remains unreliable under `gogpu/wgpu` Metal due to backend instability during repeated GPU-heavy test churn, so the dedicated Metal smoke is the trusted native verification signal for this slice.

## Phase 8: MSM

Only after scalar multiplication and G1 arithmetic are stable:

- [x] define MSM API and buffer layouts
- [x] baseline MSM using simple batching or naive decomposition
- [x] compare against gnark-crypto `MultiExp`
- [ ] measure crossover points for GPU usefulness

Later optimization candidates:

- [ ] bucket methods
- [ ] Pippenger-like decomposition
- [ ] staged reduction on GPU

Deliverable:

- initial correctness-validated MSM prototype

Notes from implementation:

- The first Phase 8 slice implements a correctness-first BN254 G1 MSM baseline in `go/curvegpu/bn254/g1_msm.go`.
- This baseline is deliberately host-orchestrated: it composes the already-validated GPU `ScalarMulAffine` and pairwise `AffineAdd` operations rather than introducing a bucketed MSM kernel too early.
- The MSM API uses a flat `count * termsPerInstance` layout for affine bases and scalar word buffers, with one affine-normalized output point per MSM instance.
- Shared vectors are generated from `gnark-crypto` in `testdata/vectors/g1/bn254_phase8_msm.json`, covering small deterministic MSMs, zero scalars, infinity bases, and `q-1`.
- A native Metal smoke command exists in `cmd/bn254-g1-msm-metal-smoke`, and it passes on Apple Silicon for `msm_naive_affine`.
- A manual headed-Chrome browser smoke exists in `web/static/bn254_g1_msm.html`, and it passes on Apple Silicon with adapter diagnostics reporting `vendor = apple` and `architecture = metal-3`.
- The Go validation for this slice lives in `go/curvegpu/bn254_g1_msm_test.go` and checks the MSM outputs against the shared vectors.
- A helper split in `shaders/curves/bn254/g1_arith.wgsl` now exposes the Jacobian-returning `g1_scalar_mul_affine_jac`, which keeps the scalar-multiplication logic reusable for future optimized MSM kernels.

## Performance track

This track intentionally deviates from the original ordering before Phase 9.

The immediate goal is no longer just primitive correctness. It is to measure whether the current BN254 primitives can be accelerated meaningfully when we lean on WebGPU-native parallel execution and use larger workloads.

This track should reuse the decomposition ideas already present in `gnark-crypto`:

- vector operations: per-slice parallel work split
- FFT / NTT: stage-oriented butterfly parallelism with threshold-based splitting
- MSM: windowed Pippenger decomposition with chunked bucket processing

Benchmark policy for this track:

- compare WebGPU Metal timings against native `gnark-crypto` Go timings
- benchmark browser WebGPU separately and compare those numbers against the native Go timings
- measure the full WebGPU path:
  - device and pipeline initialization
  - input upload
  - kernel execution
  - result readback
- start with sizes `2^10` through `2^20` when memory permits
- if a backend or device limit prevents a target size, record the largest passing size instead of silently shrinking scope
- report both correctness status and timing, not timing alone

### Performance Track 1: Parallel `fr` vector operations

- [x] extend the vector kernel for large batched elementwise `add`, `sub`, `mul`, and bit-reverse copy
- [x] add native Go benchmarks against `gnark-crypto` vector operations
- [x] add Metal WebGPU benchmarks on the same inputs and sizes
- [x] add manual headed-Chrome browser benchmarks
- [ ] measure `2^10` through `2^20` when feasible
- [ ] identify the first size where GPU becomes competitive

Deliverable:

- timing table for BN254 `fr` vector operations across native Go, Metal WebGPU, and browser WebGPU

Notes from implementation:

- The vector kernel in `shaders/curves/bn254/fr_vector.wgsl` now includes dedicated opcodes for elementwise `add` and `sub` in addition to `mul` and bit-reverse copy.
- The Go wrapper in `go/curvegpu/bn254/fr_vector.go` now exposes `Add`, `Sub`, `MulFactors`, and `BitReverseCopy`.
- The native benchmark command lives in `cmd/bn254-fr-vector-bench` and reports:
  - one-time GPU initialization cost
  - per-op `cold` and `warm` full-path timings
  - explicit `upload`, `kernel`, `readback`, and `total` timing components
  - exact output verification against `gnark-crypto`
- The manual browser benchmark page lives in `web/static/bn254_fr_vector_bench.html` and follows the same timing model.
- Interpretation note:
  - `cold_total` is the main end-to-end usefulness metric for this track.
  - In the browser, the apparent `kernel` timing is only host-side encode/submit overhead. The actual GPU wait mostly appears in `readback`, because WebGPU completion is observed at `mapAsync`.
- Initial measurements on Apple Silicon show that the current full-path WebGPU vector implementation is still slower than native `gnark-crypto` through `2^20`, even though scaling is smooth and correctness holds at each tested size.

### Performance Track 2: Parallel BN254 `fr` NTT

- [x] move from the current correctness-first stage flow to a benchmarkable large-input NTT path
- [ ] follow `gnark-crypto`'s stage-structured parallelism for butterfly work decomposition
- [x] benchmark forward and inverse NTT for power-of-two sizes starting at `2^10`
- [x] compare Metal WebGPU against native `gnark-crypto/fft`
- [x] benchmark browser WebGPU separately

Deliverable:

- timing table for BN254 `fr` NTT across native Go, Metal WebGPU, and browser WebGPU

Notes from implementation:

- The native benchmark command now lives in `cmd/bn254-fr-ntt-bench` and reports `cold` and `warm` end-to-end timings together with stage-level timing splits.
- The current Metal backend remains unstable when sweeping many large NTT sizes in a single long-lived process, so `bench-fr-ntt-range.sh` runs one benchmark process per size.
- A browser benchmark page now exists at `web/static/bn254_fr_ntt_bench.html` and follows the same full-path timing model as the vector benchmark.
- Initial measurements on Apple Silicon show the same qualitative result as the vector benchmark: the current full-path WebGPU NTT implementation is still slower than native `gnark-crypto` through the tested sizes.
- The first headed-Chrome browser run for `2^10 .. 2^14` completed successfully on the Apple Metal adapter, with forward and inverse NTT both scaling smoothly and with readback dominating the observable browser-side wait time.

### Performance Track 3: BN254 MSM with Pippenger

- [x] replace the current naive host-orchestrated MSM baseline with a correctness-first Pippenger implementation
- [ ] mirror the windowing and chunk-partition strategy used in `gnark-crypto`
- [x] benchmark the current naive MSM baseline before replacing it
- [x] benchmark MSM sizes starting at `2^10` points where practical
- [ ] compare Metal WebGPU against native `gnark-crypto MultiExp`
- [x] benchmark browser WebGPU separately
- [x] identify which substeps stay on host and which move to GPU in the first optimized design

Deliverable:

- timing table and architecture notes for BN254 MSM with a WebGPU Pippenger baseline

Notes from implementation:

- The initial Phase 8 baseline remained the host-orchestrated MSM from `go/curvegpu/bn254/g1_msm.go`, composed from GPU `ScalarMulAffine` and pairwise `AffineAdd`.
- A browser-only correctness-first Pippenger path was then added in `shaders/curves/bn254/g1_arith.wgsl`, `web/static/bn254_g1_msm.html`, and `web/static/bn254_g1_msm_bench.html`.
- The first dedicated GPU-native Pippenger attempt was incorrect in two places:
  - the extra MSM kernel parameters needed a safer uniform layout on Metal/WebGPU
  - the fast window reducer initially skipped empty-bucket running-sum contributions, which changed the effective bucket weights
- The current browser Pippenger design is now correctness-validated and uses:
  - signed sparse bucket metadata
  - chunked sparse bucket aggregation on GPU
  - parallel sparse per-window subsum on GPU
  - GPU final window combination
  - a benchmark path that keeps intermediate buffers on device and performs only one final readback
- Stable native Metal benchmarking for the optimized G1 path is still blocked by `gogpu/wgpu` instability under repeated G1 dispatch churn, so the headed-Chrome browser path is the reliable benchmark environment for this checkpoint.
- Current browser measurements on Apple Silicon now show a clear large-size win for the optimized Pippenger path:
  - `2^10`: naive `72.6 ms`, Pippenger `65.2 ms`
  - `2^11`: naive `97.8 ms`, Pippenger `63.9 ms`
  - `2^12`: naive `143.5 ms`, Pippenger `142.7 ms`
  - `2^13`: naive `281.3 ms`, Pippenger `157.0 ms`
  - `2^14`: naive `535.6 ms`, Pippenger `317.2 ms`
  - `2^15`: naive `1006.3 ms`, Pippenger `608.5 ms`
  - `2^16`: naive `1949.2 ms`, Pippenger `736.1 ms`
  - `2^17`: naive `3826.6 ms`, Pippenger `1004.9 ms`
  - `2^18`: naive `7573.7 ms`, Pippenger `1523.1 ms`
- The host-side metadata bottleneck was substantially reduced by switching the benchmark path from hex-to-`BigInt` signed-window decomposition to packed `u32` scalar words:
  - at `2^18`, `partition_ms` dropped from about `1955.8 ms` to `168.1 ms`
  - the dominant remaining costs are now GPU bucket and window reduction rather than host metadata construction

## Phase 9: Multi-curve extension

Only after BN254 primitive layers are stable:

- [ ] factor common 4-word field code for reuse by BLS12-381 / BLS12-377 scalar fields
- [ ] implement 6-word base-field backend
- [ ] add BLS12-381 `fr` / `fp` support
- [ ] add BLS12-377 `fr` / `fp` support
- [ ] evaluate whether curve-level code can share generic patterns or should remain per-curve

Deliverable:

- architecture proven to support BN254 plus at least one BLS curve without redesign

## Phase 10: Integration experiments

Only after primitives are individually validated:

- [ ] identify which prover substeps are worth offloading
- [ ] test replacing selected Groth16 subroutines with GPU primitive calls
- [ ] profile transfer overhead vs compute savings

Deliverable:

- evidence for or against practical acceleration paths

## Testing strategy

Every primitive should be testable in three ways:

1. reference CPU result from gnark-crypto
2. Metal result from Go + `gogpu/wgpu`
3. browser result from TypeScript + headed Chrome WebGPU

### Test layers

#### Unit vectors

- deterministic small cases
- edge values
- reduction boundary cases
- infinity / identity cases for curve work

#### Differential random tests

- generate random host inputs
- compute expected outputs using gnark-crypto
- run the same operation on Metal and browser
- compare exact serialized outputs

#### Cross-environment parity

- same test vectors for Go and browser
- same WGSL shader files
- same field / point serialization

### Test vector policy

The easiest way to avoid host-language drift is:

- generate test vectors in Go using gnark-crypto
- serialize them into `testdata/vectors/...`
- consume the same vectors from TypeScript and Go GPU harnesses

## Browser validation plan

Current accepted Phase 2 browser validation path:

- open `web/static/bn254_fr_ops.html` in headed Chrome
- confirm the page reports all implemented ops as `OK`
- confirm adapter diagnostics report the Apple Metal-backed adapter rather than a fallback path

Future browser automation work remains desirable, but it is now explicitly separate from primitive correctness:

- investigate a reliable automation path that works on macOS/Metal without depending on flaky headless Chrome behavior
- once stable, restore shell-script-driven browser automation for `fr`, `fp`, `g1`, and `ntt`

## Metal automation plan

Go-side tests should follow the same pattern as the current local experiment:

- dedicated Go command or tests for each primitive
- explicit GPU completion before readback
- compare against gnark-crypto in-process

Future targets:

- `make test-metal`
- `make test-metal-fr`
- `make test-metal-fp`
- `make test-metal-g1`
- `make test-metal-ntt`

## Important technical risks

### 1. `u64` to `u32` translation complexity

This is the main arithmetic risk.

Mitigation:

- use parameterized `8 x u32` and `12 x u32` limb formats from the start
- keep all operations explicit and testable
- do not optimize away reductions until correctness is established

### 1b. BN254-specific assumptions leaking into the architecture

This is the main design risk once more curves are in scope.

Mitigation:

- curve id and field id should be explicit in host APIs
- limb count should be a first-class parameter in wrappers and test harnesses
- constants and shader includes should live under curve-specific directories

### 2. WGSL layout and alignment differences

Browser WebGPU validation can be stricter than the current Go native path.

Mitigation:

- prefer storage buffers for bulk data
- use explicit padded uniform structs where needed
- maintain shared host-side layout constants

### 3. CPU/GPU synchronization bugs

We already hit one in the existing compute example.

Mitigation:

- make all readback paths explicitly wait
- keep test harnesses strict about command completion

### 4. Browser vs Metal behavior differences

The same shader may validate differently across environments.

Mitigation:

- one shader source, two host harnesses, same vectors
- browser testing is not optional

### 5. Premature optimization risk

FFT and MSM are the natural GPU targets, but they depend on a correct field layer first.

Mitigation:

- do not begin MSM optimization until field and G1 are stable
- do not begin FFT kernel fusion until a stage-by-stage version passes

## Initial implementation recommendation

The first concrete implementation step after plan approval should be:

1. create shared directories and curve-parameterized shader loading plumbing
2. define generic 4-word and 6-word WGSL field layouts
3. implement Go and TypeScript conversion wrappers for both
4. implement BN254 `fr` add / sub / neg / normalize only
5. validate against gnark-crypto on Metal and browser

Reason:

- this gives us the serialization and testing infrastructure early
- it validates both the `u32` representation choice and the multi-curve-friendly abstraction before Montgomery multiplication
- it creates the smallest useful checkpoint before more complex field logic

## Questions to confirm before implementation

These are the main decisions to confirm before coding:

- [ ] Start with `fr` first, then `fp`, then G1, then NTT, then MSM
- [ ] Use Montgomery form internally in WGSL
- [ ] Use little-endian `u32` limbs with parameterized field sizes:
  - 4-word fields -> `8 x u32`
  - 6-word fields -> `12 x u32`
- [ ] Keep a single shared WGSL source tree used by both Go and browser
- [ ] Keep the architecture curve-parameterized from the start, even though BN254 is implemented first
- [ ] Use Go-generated test vectors as the compatibility source of truth
- [ ] Keep the first milestone limited to field element representation plus basic `fr` arithmetic

## Progress notes

- [x] Existing repo compute harness works on Metal
- [x] Existing browser WebGPU harness works in headed Chrome on Apple Silicon with Metal-backed adapter diagnostics
- [x] Initial primitive implementation plan stored in repo
- [ ] Implementation not started yet
