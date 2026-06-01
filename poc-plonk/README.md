# PLONK TS Browser POC

This is the clean PLONK scaffold for BN254, BLS12-377, and BLS12-381. It mirrors
the Groth16 browser flow: JS loads gnark artifacts and witness bytes, calls
`curve.plonk`, then verifies the returned proof bytes.

Generate the initial fixtures:

```sh
make poc-plonk-fixtures
```

Build the web package and wasm runtimes:

```sh
make web-build
```

Serve the repository root and open:

```text
http://localhost:8000/poc-plonk/index.html
http://localhost:8000/poc-plonk/index.html?autorun=1&impl=native-go&curve=bn254&size-log=12&commitments=0&prove-runs=1
http://localhost:8000/poc-plonk/index.html?autorun=1&impl=webgpu-go&curve=bls12_381&size-log=18&commitments=0&prove-runs=1
```

The `webgpu-go` runtime uses the accelerated backend wrapper, including the
WebGPU quotient and commitment paths implemented by the curve-specific PLONK
packages.
