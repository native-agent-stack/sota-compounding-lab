#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SOTA_GPU_SIDECAR_PORT:-8794}"
HOST="${SOTA_GPU_SIDECAR_HOST:-127.0.0.1}"
LOG_DIR="${HOME}/sota-data-ext4/logs"
PID_FILE="${LOG_DIR}/gpu_sidecar_${PORT}.pid"
LOG_FILE="${LOG_DIR}/gpu_sidecar_${PORT}.log"

mkdir -p "${LOG_DIR}"

if [ -x "${REPO_DIR}/.venv-gpu-py312/bin/python" ]; then
  PY_BIN="${REPO_DIR}/.venv-gpu-py312/bin/python"
elif [ -x "${REPO_DIR}/.venv/bin/python" ]; then
  PY_BIN="${REPO_DIR}/.venv/bin/python"
else
  PY_BIN="$(command -v python3)"
fi

if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "http://${HOST}:${PORT}/api/gpu/health" >/dev/null 2>&1; then
  echo "[gpu-sidecar] Already healthy on http://${HOST}:${PORT}"
  exit 0
fi

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "[gpu-sidecar] Existing process PID=${OLD_PID} alive; waiting for health..."
    sleep 1
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

# Idempotent zero-copy-if-fresh sync of pre-wall lake to durable Z: ext4 tier
"${PY_BIN}" "${REPO_DIR}/scripts/sync_prewall_lake_fast_tier.py" --to-ext4 --no-sha >/dev/null 2>&1 || true

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
nohup "${PY_BIN}" -u -m sota_compounding.gpu_scoring_sidecar \
  --host "${HOST}" \
  --port "${PORT}" \
  --lake-root "/mnt/z/sota-data/prewall-lake" \
  >>"${LOG_FILE}" 2>&1 </dev/null &

NEW_PID=$!
echo "${NEW_PID}" >"${PID_FILE}"

for _ in $(seq 1 20); do
  if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "http://${HOST}:${PORT}/api/gpu/health" >/dev/null 2>&1; then
    echo "[gpu-sidecar] Started PID=${NEW_PID} healthy on http://${HOST}:${PORT}"
    exit 0
  fi
  sleep 0.25
done

echo "[gpu-sidecar] Started PID=${NEW_PID} (check ${LOG_FILE} if slow warm-up)"
