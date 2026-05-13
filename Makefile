.PHONY: web-build web-bundle-shaders web-lint web-groth16-assets testdata fixture-bn254-g1 fixture-bls12_377-g1 fixture-bls12_381-g1 fixture-bn254-g2 fixture-bls12_377-g2 fixture-bls12_381-g2 poc-gnark-groth16-fixtures

COUNT ?= 524288
ITERS ?= 1
FIXTURE_CURVE ?= all
FIXTURE_LOGS ?= 12,15,18
FIXTURE_COMMITMENTS ?= 0,1,2

web-bundle-shaders:
	cd web && npm run build:shaders

web-lint:
	cd web && npm run lint

web-build: web-bundle-shaders web-lint
	cd web && npm run build
	$(MAKE) web-groth16-assets

web-groth16-assets:
	mkdir -p web/dist/assets
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" web/dist/assets/wasm_exec.js
	GOOS=js GOARCH=wasm go build -o web/dist/assets/groth16-webgpu.wasm ./backend/accelerated/webgpu/groth16/internal/wasmruntime/webgpu
	GOOS=js GOARCH=wasm go build -o web/dist/assets/groth16-native.wasm ./backend/accelerated/webgpu/groth16/internal/wasmruntime/native

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

poc-gnark-groth16-fixtures:
	go run ./cmd/poc-gnark-groth16-fixtures -curve $(FIXTURE_CURVE) -logs $(FIXTURE_LOGS) -commitments $(FIXTURE_COMMITMENTS)
