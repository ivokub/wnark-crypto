.PHONY: web-build testdata fixture-bn254-g1 fixture-bls12_381-g1 metal-bench-fr-ntt-range

COUNT ?= 524288
MIN_LOG ?= 10
MAX_LOG ?= 20
ITERS ?= 1

web-build:
	cd web && npm run build

testdata:
	go generate ./testdata

fixture-bn254-g1:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g1-bases-fixture -fixture-count $(COUNT)

fixture-bls12_381-g1:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g1-bases-fixture -fixture-count $(COUNT)

metal-bench-fr-ntt-range:
	@for log in $$(seq $(MIN_LOG) $(MAX_LOG)); do \
		CGO_ENABLED=0 go run ./internal/metal/cmd/benchutil -curve bn254 -suite fr_ntt -min-log $$log -max-log $$log -iters $(ITERS); \
	done
