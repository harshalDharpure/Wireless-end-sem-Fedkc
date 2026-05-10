#!/usr/bin/env bash
# Supervisor that keeps the sweep running across crashes.
# Resume logic in src.experiments ensures no completed run is repeated.
set -u

cd "$(dirname "$0")/.." || exit 1

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR" /tmp/ray_fedlearn

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
SWEEP_CFG="${SWEEP_CFG:-configs/sweep.yaml}"
SUP_LOG="${LOG_DIR}/supervisor.log"
SWEEP_LOG="results/sweep.log"

ATTEMPT=0
MAX_ATTEMPTS=200    # plenty of headroom; resume keeps work bounded
SLEEP_BETWEEN=10    # seconds before relaunch

ts() { date +%F_%H:%M:%S; }
log() { echo "[supervisor $(ts)] $*" | tee -a "$SUP_LOG" >&2; }

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  log "attempt #$ATTEMPT  CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS  sweep=$SWEEP_CFG"

  # Rotate sweep.log on every (re)launch so progress is easy to see.
  if [ -f "$SWEEP_LOG" ]; then
    mv "$SWEEP_LOG" "${SWEEP_LOG}.attempt_${ATTEMPT}_$(date +%Y%m%d_%H%M%S)"
  fi

  CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python -u -m src.experiments --sweep "$SWEEP_CFG" \
    > "$SWEEP_LOG" 2>&1
  RC=$?
  log "sweep exited rc=$RC after attempt #$ATTEMPT"

  if grep -q '^Sweep finished:' "$SWEEP_LOG" 2>/dev/null; then
    log "Detected 'Sweep finished:' — done. Bye."
    exit 0
  fi

  log "sleeping ${SLEEP_BETWEEN}s before relaunch"
  sleep "$SLEEP_BETWEEN"
done

log "max attempts reached ($MAX_ATTEMPTS); giving up"
exit 1
