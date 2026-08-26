#!/usr/bin/env bash
# Launch the remaining multi-seed replicate work, detached and sleep-guarded.
#
#   jobs/run_tonight.sh
#
# Runs every chain SERIALLY under a single caffeinate. Completed stages are
# skipped automatically, so this is safe to re-run after an interruption.
# Check progress any time with: jobs/status.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
mkdir -p jobs/logs

if pgrep -f "run_seed_chain|run_chains_serial" >/dev/null; then
  echo "FL runs already in progress. Use jobs/status.sh, or kill them first."; exit 1
fi

: "${FL_NUM_THREADS:=3}"
: "${FL_PORT:=8081}"
export FL_NUM_THREADS FL_PORT

STAMP="$(date '+%m%d_%H%M')"
LOG="jobs/logs/serial_${STAMP}.log"

# One caffeinate, one chain resident at a time. -i prevents idle sleep, -m
# keeps the disk awake, -s prevents system sleep on AC.
nohup caffeinate -ims jobs/run_chains_serial.sh > "$LOG" 2>&1 &
disown

sleep 5
echo "Launched serially, threads=$FL_NUM_THREADS per process, port=$FL_PORT"
echo "Log: $LOG"

# Pair a stall watchdog with the batch. Idempotent: if one is already running
# (for instance because the watchdog itself triggered this relaunch), it says
# so and does nothing.
jobs/start_watchdog.sh "$LOG" || echo "WARNING: continuing without a watchdog."
echo
echo "  chain 1  seed 7   async K=2        ~5h"
echo "  chain 2  seed 13  sync then async  ~10h"
echo
echo "Check with: jobs/status.sh"
echo "Tail with:  tail -f $LOG"
