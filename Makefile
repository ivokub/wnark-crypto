.PHONY: web-build testdata fixture-bn254-g1 fixture-bls12_381-g1 browser-suite browser-smoke browser-bench

COUNT ?= 524288
CURVE ?= bn254
SUITE ?= fr_ops
MIN_LOG ?=
MAX_LOG ?=
ITERS ?=
BASE_SOURCE ?=
EXTRA_QUERY ?=
BROWSER_MODE ?= headless
PORT ?= 8000
VIRTUAL_TIME_BUDGET ?= 180000
CHROME_BIN ?=

web-build:
	cd web && npm run build

testdata:
	go generate ./testdata

fixture-bn254-g1:
	go run ./cmd/curvegpu-testdata-gen -target bn254-g1-bases-fixture -fixture-count $(COUNT)

fixture-bls12_381-g1:
	go run ./cmd/curvegpu-testdata-gen -target bls12-381-g1-bases-fixture -fixture-count $(COUNT)

browser-suite:
	PORT=$(PORT) \
	CURVE=$(CURVE) \
	SUITE=$(SUITE) \
	MIN_LOG=$(MIN_LOG) \
	MAX_LOG=$(MAX_LOG) \
	ITERS=$(ITERS) \
	BASE_SOURCE=$(BASE_SOURCE) \
	EXTRA_QUERY='$(EXTRA_QUERY)' \
	MODE=$(BROWSER_MODE) \
	VIRTUAL_TIME_BUDGET=$(VIRTUAL_TIME_BUDGET) \
	CHROME_BIN='$(CHROME_BIN)' \
	bash ./web/tests/browser/run-suite.sh

browser-smoke: browser-suite

browser-bench: browser-suite
