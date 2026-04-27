# Groth16 Go WASM POC

This POC compares two browser-side wasm Groth16 proof paths on the same fixed
no-commitment circuit fixtures for BN254, BLS12-377, and BLS12-381:

- `webgpu-go`: the local WebGPU-accelerated prover package in this repo
- `native-go`: native gnark Groth16 compiled to wasm

The benchmark flow is:

1. load serialized `ccs`, `pk`, and `vk` fixture files,
2. build the witness in wasm,
3. generate a proof,
4. serialize it with `WriteTo`,
5. deserialize it with `ReadFrom`,
6. verify it with gnark `groth16.Verify`.

Proof bytes are not compared because Groth16 proving is randomized.

## Fixture Generation

The fixture artifacts are local generated data and are ignored by git. Generate them with:

```sh
make poc-gnark-groth16-fixtures
```

This creates:

- `poc-gnark-groth16/fixtures/bn254/2pow12/`
- `poc-gnark-groth16/fixtures/bn254/2pow15/`
- `poc-gnark-groth16/fixtures/bn254/2pow18/`
- `poc-gnark-groth16/fixtures/bls12_377/2pow12/`
- `poc-gnark-groth16/fixtures/bls12_377/2pow15/`
- `poc-gnark-groth16/fixtures/bls12_377/2pow18/`
- `poc-gnark-groth16/fixtures/bls12_381/2pow12/`
- `poc-gnark-groth16/fixtures/bls12_381/2pow15/`
- `poc-gnark-groth16/fixtures/bls12_381/2pow18/`

Each fixture directory contains:

- `ccs.bin`
- `pk.dump`
- `vk.bin`

`pk.dump` uses gnark's fast unsafe proving-key dump format. The browser loader
falls back to legacy `pk.bin` fixtures if they already exist locally.

You can scope generation, for example:

```sh
make poc-gnark-groth16-fixtures FIXTURE_CURVE=bls12_377 FIXTURE_LOGS=12,15
```

## Build

```sh
make poc-gnark-groth16-build
```

This will:

- build the browser library under `web/dist/`
- compile the two Go wasm binaries into `poc-gnark-groth16/dist/`
- copy `wasm_exec.js` into `poc-gnark-groth16/dist/`

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
http://localhost:8000/poc-gnark-groth16/index.html?autorun=1&impl=both&curve=bls12_377&size-log=15&prove-runs=1
http://localhost:8000/poc-gnark-groth16/index.html?autorun=1&impl=both&curve=bls12_381&size-log=15&prove-runs=1
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
