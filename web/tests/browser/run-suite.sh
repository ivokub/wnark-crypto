#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${PORT:-8000}"
CURVE="${CURVE:-bn254}"
SUITE="${SUITE:-fr_ops}"
MODE="${MODE:-headless}"
VIRTUAL_TIME_BUDGET="${VIRTUAL_TIME_BUDGET:-180000}"
MIN_LOG="${MIN_LOG:-}"
MAX_LOG="${MAX_LOG:-}"
ITERS="${ITERS:-}"
BASE_SOURCE="${BASE_SOURCE:-}"
EXTRA_QUERY="${EXTRA_QUERY:-}"
SERVER_PID=""
SERVER_LOG=""
PROFILE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/curvegpu-chrome-profile.XXXXXX")"
DOM_FILE=""

default_chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ -z "${CHROME_BIN:-}" ]]; then
  if [[ -x "${default_chrome}" ]]; then
    CHROME_BIN="${default_chrome}"
  elif command -v google-chrome >/dev/null 2>&1; then
    CHROME_BIN="$(command -v google-chrome)"
  elif command -v chromium >/dev/null 2>&1; then
    CHROME_BIN="$(command -v chromium)"
  elif command -v chromium-browser >/dev/null 2>&1; then
    CHROME_BIN="$(command -v chromium-browser)"
  else
    echo "unable to find Chrome/Chromium; set CHROME_BIN explicitly" >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DOM_FILE}" && -f "${DOM_FILE}" ]]; then
    rm -f "${DOM_FILE}"
  fi
  if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
    rm -f "${SERVER_LOG}"
  fi
  rm -rf "${PROFILE_DIR}"
}
trap cleanup EXIT

server_url="http://127.0.0.1:${PORT}"
harness_url="${server_url}/web/tests/browser/curvegpu.html"
if ! curl -fsS "${harness_url}" >/dev/null 2>&1; then
  SERVER_LOG="$(mktemp "${TMPDIR:-/tmp}/curvegpu-http.XXXXXX.log")"
  (
    cd "${ROOT_DIR}"
    python3 -m http.server "${PORT}" --bind 127.0.0.1 >"${SERVER_LOG}" 2>&1
  ) &
  SERVER_PID=$!
  for _ in $(seq 1 50); do
    if curl -fsS "${harness_url}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  if ! curl -fsS "${harness_url}" >/dev/null 2>&1; then
    echo "failed to start local HTTP server on ${server_url}" >&2
    if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
      cat "${SERVER_LOG}" >&2
    fi
    exit 1
  fi
fi

query="curve=${CURVE}&suite=${SUITE}&autorun=1"
if [[ -n "${MIN_LOG}" ]]; then
  query="${query}&min-log=${MIN_LOG}"
fi
if [[ -n "${MAX_LOG}" ]]; then
  query="${query}&max-log=${MAX_LOG}"
fi
if [[ -n "${ITERS}" ]]; then
  query="${query}&iters=${ITERS}"
fi
if [[ -n "${BASE_SOURCE}" ]]; then
  query="${query}&base-source=${BASE_SOURCE}"
fi
if [[ -n "${EXTRA_QUERY}" ]]; then
  query="${query}&${EXTRA_QUERY}"
fi

url="${server_url}/web/tests/browser/curvegpu.html?${query}"

chrome_args=(
  --enable-unsafe-webgpu
  --use-angle=metal
  --user-data-dir="${PROFILE_DIR}"
  --no-first-run
  --no-default-browser-check
  --disable-background-networking
  --disable-component-update
  --disable-default-apps
  --disable-sync
  --metrics-recording-only
  --password-store=basic
  --use-mock-keychain
)

if [[ "${MODE}" == "headed" ]]; then
  "${CHROME_BIN}" "${chrome_args[@]}" "${url}"
  exit 0
fi

DOM_FILE="$(mktemp "${TMPDIR:-/tmp}/curvegpu-dom.XXXXXX.html")"
"${CHROME_BIN}" \
  "${chrome_args[@]}" \
  --headless=new \
  --virtual-time-budget="${VIRTUAL_TIME_BUDGET}" \
  --dump-dom \
  "${url}" >"${DOM_FILE}"

cat "${DOM_FILE}"

if grep -q 'data-status="pass"' "${DOM_FILE}"; then
  exit 0
fi
if grep -q 'data-status="fail"' "${DOM_FILE}"; then
  exit 1
fi

echo "browser suite did not reach a pass/fail terminal state" >&2
exit 1
