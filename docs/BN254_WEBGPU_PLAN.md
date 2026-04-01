# Pairing-Friendly Curve WebGPU Primitive Plan

Status: Draft for review
Last updated: 2026-04-01

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
- [ ] Confirm plan and initial scope with user
- [ ] Create directory layout for shaders, Go host code, TypeScript host code, and testdata
- [ ] Make the scaffolding curve-parameterized even if only BN254 is implemented first
- [ ] Add shared shader loading strategy for Go and TypeScript
- [ ] Add browser and Metal test harness skeletons for future kernels

Deliverable:

- repo structure ready for primitive work, but no arithmetic yet

## Phase 1: Shared representation and conversion layer

- [ ] Define WGSL field element structs for 4-word and 6-word fields
- [ ] Define Go wrapper types for WGSL buffer-compatible field elements
- [ ] Define TypeScript wrapper types for WGSL buffer-compatible field elements
- [ ] Implement Go conversion: `[4]uint64` <-> `[8]u32`
- [ ] Implement Go conversion: `[6]uint64` <-> `[12]u32`
- [ ] Implement TS conversion: bytes / bigint / limb arrays <-> GPU limbs
- [ ] Define canonical serialization policy for tests
- [ ] Add golden test vectors for edge cases:
  - zero
  - one
  - modulus minus one
  - random values
  - carry-heavy values

Deliverable:

- host can move BN254 field elements between gnark-crypto and WGSL-compatible buffers without ambiguity
- the shared representation layer is ready for future BLS12-381 / BLS12-377 support

## Phase 2: `fr` arithmetic baseline

First field to implement: `fr`.

Operations:

- [ ] zero / one / copy / equality
- [ ] addition
- [ ] subtraction
- [ ] negation
- [ ] conditional subtraction / normalization
- [ ] doubling
- [ ] Montgomery multiplication
- [ ] squaring
- [ ] conversion to Montgomery
- [ ] conversion from Montgomery

Kernel strategy:

- [ ] single-op kernels for vectorized elementwise testing
- [ ] tiny smoke kernels for deterministic examples

Tests:

- [ ] Go CPU vs GPU random differential tests against gnark-crypto `fr`
- [ ] Browser CPU vs GPU random differential tests against exported test vectors
- [ ] Metal/browser parity tests on the same vectors
- [ ] edge-case tests around carries and reduction boundaries

Deliverable:

- stable `fr` arithmetic layer verified in Metal and browser

## Phase 3: `fp` arithmetic baseline

Implement the same operation set as `fr`, but over `fp`.

- [ ] replicate WGSL field logic with `fp` constants
- [ ] host wrappers for `fp`
- [ ] differential tests against gnark-crypto `fp`
- [ ] shared reducer and carry helpers where possible

Deliverable:

- stable `fp` arithmetic layer verified in Metal and browser

## Phase 4: `fr` vector operations and basic kernels

Once scalar operations are stable:

- [ ] vector add / sub / mul kernels
- [ ] batched Montgomery conversions
- [ ] batched scalar multiply by constants
- [ ] permutation / copy / bit-reversal helpers relevant to FFT

Deliverable:

- enough array-oriented `fr` support to support FFT experiments

## Phase 5: FFT / NTT over `fr`

Initial goal:

- correctness-first radix-2 power-of-two NTT, matching gnark-crypto domain logic

Tasks:

- [ ] mirror domain generation and twiddle handling from gnark-crypto
- [ ] choose first implementation shape:
  - simple iterative DIT/DIF
  - explicit stage-by-stage kernels
- [ ] implement bit-reversal strategy
- [ ] implement forward NTT
- [ ] implement inverse NTT
- [ ] verify on multiple domain sizes
- [ ] compare against gnark-crypto FFT outputs

Testing:

- [ ] small exact cases
- [ ] random vectors
- [ ] coset and inverse cases if needed later

Deliverable:

- browser and Metal NTT compatible with gnark-crypto for supported domain sizes

## Phase 6: G1 point representation and basic arithmetic

Initial curve scope should be G1 only.

Representations to support:

- [ ] affine
- [ ] Jacobian

Operations:

- [ ] affine infinity encoding policy
- [ ] affine/jacobian conversion
- [ ] point equality
- [ ] affine add
- [ ] mixed add
- [ ] Jacobian double
- [ ] negation
- [ ] on-curve checks in host-side tests

Testing:

- [ ] compare with gnark-crypto G1 outputs
- [ ] include infinity and same-point cases
- [ ] include `P + (-P)` and doubling corner cases

Deliverable:

- correctness-first G1 arithmetic baseline

## Phase 7: Scalar multiplication

Start with the simplest correct algorithm first.

- [ ] fixed-base scalar multiplication baseline
- [ ] variable-base double-and-add baseline
- [ ] batched scalar multiplication harness
- [ ] compare against gnark-crypto `ScalarMultiplication`

Later optimization candidates:

- [ ] windowed scalar multiplication
- [ ] mixed coordinate scheduling
- [ ] GLV decomposition

Deliverable:

- correct scalar multiplication before optimizing MSM

## Phase 8: MSM

Only after scalar multiplication and G1 arithmetic are stable:

- [ ] define MSM API and buffer layouts
- [ ] baseline MSM using simple batching or naive decomposition
- [ ] compare against gnark-crypto `MultiExp`
- [ ] measure crossover points for GPU usefulness

Later optimization candidates:

- [ ] bucket methods
- [ ] Pippenger-like decomposition
- [ ] staged reduction on GPU

Deliverable:

- initial correctness-validated MSM prototype

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
3. browser result from TypeScript + headless Chrome

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

## Browser automation plan

We already have a headless Chrome pattern working in this repo. We should extend the same approach:

- TypeScript/browser page supports `autorun=1`
- page writes `data-status="pass|fail"`
- page logs operation results and mismatches to DOM
- shell script / test target runs headless Chrome and checks the DOM

Future targets:

- `make test-browser`
- `make test-browser-fr`
- `make test-browser-fp`
- `make test-browser-g1`
- `make test-browser-ntt`

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
- [x] Existing browser WebGPU harness works under headless Chrome
- [x] Initial primitive implementation plan stored in repo
- [ ] Implementation not started yet
