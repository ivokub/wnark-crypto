#!/usr/bin/env bash
set -euo pipefail

MIN_LOG="${MIN_LOG:-10}"
MAX_LOG="${MAX_LOG:-20}"
ITERS="${ITERS:-1}"

for ((log = MIN_LOG; log <= MAX_LOG; log++)); do
  CGO_ENABLED=0 go run ./cmd/curvegpu-native-bench -curve bn254 -suite fr_ntt -min-log "$log" -max-log "$log" -iters "$ITERS"
done
