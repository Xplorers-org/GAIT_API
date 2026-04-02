#!/usr/bin/env sh
set -eu

PORT="${PORT:-7860}"
CLEANUP_INTERVAL_SECONDS="${CLEANUP_INTERVAL_SECONDS:-1800}"
RUNS_MAX_AGE_MINUTES="${RUNS_MAX_AGE_MINUTES:-30}"

mkdir -p /app/runs/outputs

cleanup_loop() {
  while true; do
    python /app/scripts/cleanup_runs.py \
      --path /app/runs \
      --max-age-minutes "${RUNS_MAX_AGE_MINUTES}" || true
    sleep "${CLEANUP_INTERVAL_SECONDS}"
  done
}

cleanup_loop &

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"
