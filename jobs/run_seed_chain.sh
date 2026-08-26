#!/usr/bin/env bash
# Run a multi-seed replicate chain: synchronous FedAvg baseline, then async K.
#
# Usage:
#   FL_SEED=7  FL_K=2 FL_PORT=8081 FL_STAGES=async jobs/run_seed_chain.sh
#   FL_SEED=13 FL_K=2 FL_PORT=8082 FL_STAGES=both  jobs/run_seed_chain.sh
#
# FL_STAGES: both (default) | sync | async
#
# Deliberately does NOT use `set -e`. run_fl.py exits non-zero even on success
# because Flower's gRPC teardown raises in sys.unraisablehook at interpreter
# shutdown. Stage success is therefore judged by the presence of the final
# round's metrics file, not by exit code.
set -uo pipefail

: "${FL_SEED:?FL_SEED is required}"
: "${FL_K:=2}"
: "${FL_PORT:=8080}"
: "${FL_STAGES:=both}"

# Cap PyTorch intra-op threads per process. A run launches 4 processes and
# each one otherwise defaults to the full core count (10 here), so the machine
# ends up 4x oversubscribed. Verified bitwise identical at 1, 2, 3, 4, 8 and 10
# threads (jobs/precheck_threads.py, 2026-08-20), so this is numerically free.
: "${FL_NUM_THREADS:=3}"

export FL_SEED FL_K FL_PORT FL_NUM_THREADS

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/venv/bin/python"

SYNC_DONE="results/sync-federated_s${FL_SEED}/sync_IID_4R_3C_1L_s${FL_SEED}/checkpoints/metrics_round_4.json"
ASYNC_DONE="results/async-federated/async_IID_4R_3C_1L_K${FL_K}_s${FL_SEED}/checkpoints/metrics_round_4.json"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] seed=$FL_SEED K=$FL_K port=$FL_PORT threads=$FL_NUM_THREADS $*"; }

run_stage() {  # $1 = label, $2 = script, $3 = expected artifact
  local label="$1" script="$2" artifact="$3"
  local run_dir; run_dir="$(dirname "$(dirname "$artifact")")"

  if [ -f "$artifact" ]; then
    if "$PY" jobs/validate_run.py "$run_dir" >/dev/null 2>&1; then
      log "$label: already complete and valid, skipping"
      return 0
    fi
    log "$label: previous result is INVALID, discarding and re-running"
  fi

  # Archive any partial output from an interrupted earlier attempt before
  # restarting. checkpoints/all_metrics.json is written by loading the
  # existing list and appending to it, so a restart into a dirty directory
  # yields a duplicated round series such as [0,1,2,0,1,2,3,4], which
  # silently corrupts the training curve and anything that parses it.
  if [ -d "$run_dir" ] && [ -n "$(ls -A "$run_dir" 2>/dev/null)" ]; then
    local stash="results/_partial_$(date '+%Y%m%d_%H%M%S')_$(basename "$run_dir")"
    mkdir -p "$(dirname "$stash")"
    mv "$run_dir" "$stash"
    log "$label: archived incomplete previous attempt to $stash"
  fi

  log "$label: starting"
  "$PY" "$script"
  local rc=$?

  if [ ! -f "$artifact" ]; then
    log "$label: FAILED (exit $rc, no $artifact)"
    return 1
  fi

  # The artifact existing is necessary but not sufficient. A run whose clients
  # died mid-way still writes a final metrics file and still reports an
  # accuracy, which is more dangerous than an obvious crash because the number
  # looks usable. Seed 7 async K=2 did exactly this on 2026-08-21: two of three
  # clients died of gRPC "Socket closed" after round 2 and the last two rounds
  # ran on a single client.
  if "$PY" jobs/validate_run.py "$run_dir" >/dev/null 2>&1; then
    log "$label: COMPLETE (exit $rc ignored; artifact present and validated)"
    return 0
  fi

  log "$label: INVALID despite artifact present:"
  "$PY" jobs/validate_run.py "$run_dir" 2>&1 | while IFS= read -r line; do log "      $line"; done
  local stash="results/_invalid_$(date '+%Y%m%d_%H%M%S')_$(basename "$run_dir")"
  mkdir -p "$(dirname "$stash")"
  mv "$run_dir" "$stash"
  log "$label: quarantined to $stash"
  return 1
}

log "chain start (stages=$FL_STAGES)"

if [ "$FL_STAGES" = "both" ] || [ "$FL_STAGES" = "sync" ]; then
  run_stage "stage sync" federated/synchronous/run_fl.py "$SYNC_DONE" || { log "chain aborted"; exit 1; }
fi

if [ "$FL_STAGES" = "both" ] || [ "$FL_STAGES" = "async" ]; then
  if [ ! -f "$SYNC_DONE" ]; then
    log "cannot run async: sync partitions/results missing for seed $FL_SEED"; exit 1
  fi
  run_stage "stage async K=$FL_K" federated/asynchronous/run_fl.py "$ASYNC_DONE" || { log "chain aborted"; exit 1; }
fi

log "chain done"
