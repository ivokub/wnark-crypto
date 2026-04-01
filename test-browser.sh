#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}/example.html?autorun=1"
CHROME_BIN="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
VIRTUAL_TIME_BUDGET="${VIRTUAL_TIME_BUDGET:-30000}"

if [[ ! -x "${CHROME_BIN}" ]]; then
  echo "Chrome not found at: ${CHROME_BIN}" >&2
  exit 1
fi

SERVER_PID=""
SERVER_LOG="$(mktemp)"
DOM_OUT="$(mktemp)"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${SERVER_LOG}" "${DOM_OUT}"
}
trap cleanup EXIT

cd "${ROOT_DIR}"
python3 -m http.server "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 50); do
  if python3 - <<'PY' >/dev/null 2>&1
import socket
s = socket.socket()
try:
    s.connect(("127.0.0.1", int(__import__("os").environ["PORT"])))
finally:
    s.close()
PY
  then
    break
  fi
  sleep 0.1
done

for _ in 1 2 3; do
  "${CHROME_BIN}" \
    --headless=new \
    --enable-unsafe-webgpu \
    --use-angle=metal \
    --virtual-time-budget="${VIRTUAL_TIME_BUDGET}" \
    --dump-dom \
    "${URL}" >"${DOM_OUT}"

  if grep -q 'data-status="pass"' "${DOM_OUT}"; then
    break
  fi

  if grep -q 'data-status="running"' "${DOM_OUT}"; then
    sleep 1
    continue
  fi

  echo "Browser WebGPU test failed." >&2
  cat "${DOM_OUT}" >&2
  exit 1
done

if ! grep -q 'data-status="pass"' "${DOM_OUT}"; then
  echo "Browser WebGPU test did not finish in time." >&2
  cat "${DOM_OUT}" >&2
  exit 1
fi

python3 - "${DOM_OUT}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r'<pre id="log">(.*?)</pre>', text, re.S)
if not match:
    print("PASS marker found, but log block was not found.")
    sys.exit(0)

log = match.group(1)
log = (log.replace("&lt;", "<")
          .replace("&gt;", ">")
          .replace("&amp;", "&"))
print(log)
PY
