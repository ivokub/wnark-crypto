# PLONK TS Browser POC

This is the clean PLONK scaffold. It is BN254-only for now and mirrors the
Groth16 browser flow: JS loads gnark artifacts and witness bytes, calls
`curve.plonk`, then verifies the returned proof bytes.

Generate the initial fixtures:

```sh
make poc-plonk-fixtures
make poc-plonk-fixtures PLONK_FIXTURE_COMMITMENTS=1,2
```

Build the web package and wasm runtimes:

```sh
make web-build
```

Serve the repository root and open:

```text
http://localhost:8000/poc-plonk/index.html
http://localhost:8000/poc-plonk/index.html?autorun=1&impl=native-go&curve=bn254&size-log=12&commitments=0&prove-runs=1
```

The `webgpu-go` runtime currently uses the accelerated backend wrapper, but the
wrapper still delegates proving to gnark's native BN254 PLONK prover. This is
intentional scaffolding for replacing phases with WebGPU kernels incrementally.
