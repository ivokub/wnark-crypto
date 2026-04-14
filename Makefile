.PHONY: web-build web-bundle-shaders testdata fixture-bn254-g1 fixture-bls12_381-g1 fixture-bn254-g2 fixture-bls12_381-g2 metal-bench-fr-ntt-range poc-wasm-build poc-wasm-fixture-bn254 poc-wasm-fixture-bls12_381 poc-wasm-fixture-bn254-2pow18 poc-wasm-fixture-bls12_381-2pow18

COUNT ?= 524288
G2_COUNT ?= 524288
MIN_LOG ?= 10
MAX_LOG ?= 20
ITERS ?= 1

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

fixture-bn254-g2:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g2-bases-fixture -g2-fixture-count $(G2_COUNT)

fixture-bls12_381-g2:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g2-bases-fixture -g2-fixture-count $(G2_COUNT)

metal-bench-fr-ntt-range:
	@for log in $$(seq $(MIN_LOG) $(MAX_LOG)); do \
		CGO_ENABLED=0 go run ./internal/metal/cmd/benchutil -curve bn254 -suite fr_ntt -min-log $$log -max-log $$log -iters $(ITERS); \
	done

poc-wasm-build: web-build
	mkdir -p poc-wasm/dist
	cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" poc-wasm/dist/wasm_exec.js
	GOCACHE=$$(pwd)/.gocache GOOS=js GOARCH=wasm go build -o poc-wasm/dist/go-webgpu.wasm ./poc-wasm/go-webgpu
	GOCACHE=$$(pwd)/.gocache GOOS=js GOARCH=wasm go build -o poc-wasm/dist/go-gnark.wasm ./poc-wasm/go-gnark

poc-wasm-fixture-bn254:
	$(MAKE) fixture-bn254-g1 COUNT=$(COUNT)

poc-wasm-fixture-bls12_381:
	$(MAKE) fixture-bls12_381-g1 COUNT=$(COUNT)

poc-wasm-fixture-bn254-2pow18:
	$(MAKE) poc-wasm-fixture-bn254 COUNT=262144

poc-wasm-fixture-bls12_381-2pow18:
	$(MAKE) poc-wasm-fixture-bls12_381 COUNT=262144
