.PHONY: web-build web-bundle-shaders testdata fixture-bn254-g1 fixture-bls12_377-g1 fixture-bls12_381-g1 fixture-bn254-g2 fixture-bls12_377-g2 fixture-bls12_381-g2 poc-gnark-groth16-build poc-gnark-groth16-fixtures

COUNT ?= 524288
ITERS ?= 1
FIXTURE_CURVE ?= all
FIXTURE_LOGS ?= 12,15,18

web-bundle-shaders:
	cd web && npm run build:shaders

web-build: web-bundle-shaders
	cd web && npm run build

testdata:
	go generate ./testdata

fixture-bn254-g1:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g1-bases-fixture -fixture-count $(COUNT)

fixture-bls12_381-g1:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g1-bases-fixture -fixture-count $(COUNT)

fixture-bls12_377-g1:
	go run ./cmd/curvegpu-testdata-gen -target bls12-377-g1-bases-fixture -fixture-count $(COUNT)

fixture-bn254-g2:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g2-bases-fixture -g2-fixture-count $(COUNT)

fixture-bls12_377-g2:
	go run ./cmd/curvegpu-testdata-gen -target bls12-377-g2-bases-fixture -g2-fixture-count $(COUNT)

fixture-bls12_381-g2:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g2-bases-fixture -g2-fixture-count $(COUNT)

poc-gnark-groth16-build: web-build
	mkdir -p poc-gnark-groth16/dist
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" poc-gnark-groth16/dist/wasm_exec.js
	GOOS=js GOARCH=wasm go build -o poc-gnark-groth16/dist/go-webgpu.wasm ./poc-gnark-groth16/go-webgpu
	GOOS=js GOARCH=wasm go build -o poc-gnark-groth16/dist/go-native.wasm ./poc-gnark-groth16/go-native

poc-gnark-groth16-fixtures:
	go run ./cmd/poc-gnark-groth16-fixtures -curve $(FIXTURE_CURVE) -logs $(FIXTURE_LOGS)
