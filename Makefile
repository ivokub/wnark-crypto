.PHONY: web-build testdata fixture-bn254-g1 fixture-bls12_381-g1

COUNT ?= 524288

web-build:
	cd web && npm run build

testdata:
	go generate ./testdata

fixture-bn254-g1:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g1-bases-fixture -fixture-count $(COUNT)

fixture-bls12_381-g1:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g1-bases-fixture -fixture-count $(COUNT)
