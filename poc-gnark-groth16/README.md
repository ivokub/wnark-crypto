# Groth16 Go WASM POC

This POC compares two browser-side wasm Groth16 proof paths on the same fixed
circuit fixtures for BN254, BLS12-377, and BLS12-381. Each fixture has a
commitment variant:

- `commit0`: no BSB22 commitments
- `commit1`: one commitment over the first quarter of private chain variables
- `commit2`: two commitments over two disjoint quarters of private chain variables

- `webgpu-go`: the local WebGPU-accelerated prover package in this repo
- `native-go`: native gnark Groth16 compiled to wasm

The benchmark flow is now controlled from JavaScript:

1. load serialized `ccs`, `pk`, and `vk` fixture files,
2. construct gnark binary witness bytes from deterministic `bigint` values,
3. pass fixture and witness bytes to the TS-facing Groth16 module,
4. generate serialized proof bytes,
5. verify the proof bytes through the same TS-facing module.

Proof bytes are not compared because Groth16 proving is randomized.

## Fixture Generation

The fixture artifacts are local generated data and are ignored by git. Generate them with:

```sh
make poc-gnark-groth16-fixtures
```

This creates:

- `poc-gnark-groth16/fixtures/bn254/2pow12/commit0/`
- `poc-gnark-groth16/fixtures/bn254/2pow12/commit1/`
- `poc-gnark-groth16/fixtures/bn254/2pow12/commit2/`
- the same `commit0`, `commit1`, and `commit2` layout for `2pow15`, `2pow18`,
  `bls12_377`, and `bls12_381`

Each fixture directory contains:

- `ccs.bin`
- `pk.dump`
- `vk.bin`

`pk.dump` uses gnark's fast unsafe proving-key dump format.

You can scope generation, for example:

```sh
make poc-gnark-groth16-fixtures FIXTURE_CURVE=bls12_377 FIXTURE_LOGS=12,15
make poc-gnark-groth16-fixtures FIXTURE_CURVE=bn254 FIXTURE_LOGS=12 FIXTURE_COMMITMENTS=1,2
```

## Build

```sh
make web-build
```

This will:

- build the browser library under `web/dist/`
- compile the Go wasm Groth16 runtimes into `web/dist/assets/`

The POC uses the default runtime URLs exported by the TS library, so it loads
`wasm_exec.js` and the Groth16 wasm runtimes directly from `web/dist/assets/`.

## Run

Serve the repo root, for example:

```sh
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/poc-gnark-groth16/index.html
```

Example autorun URLs:

```text
http://localhost:8000/poc-gnark-groth16/index.html?autorun=1&impl=both&curve=bn254&size-log=12&prove-runs=1
http://localhost:8000/poc-gnark-groth16/index.html?autorun=1&impl=both&curve=bls12_377&size-log=15&commitments=1&prove-runs=1
http://localhost:8000/poc-gnark-groth16/index.html?autorun=1&impl=both&curve=bls12_381&size-log=15&commitments=2&prove-runs=1
```

Useful outputs:

- `fixture_load_ms`
- `witness_build_ms`
- `prepare_ms` on the WebGPU path
- `prove_round_*_ms`
- `proof_size_bytes`
- `roundtrip_verify_round_* = OK`
- `steady_state_total_ms`
- `overall_total_ms`
