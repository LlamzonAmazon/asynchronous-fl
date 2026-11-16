#!/usr/bin/env bash
# Launch the replicate batch detached, with a host-sleep guard and the stall watchdog.
#
#   jobs/launch_batch.sh
#
# Runs every chain serially under a single caffeinate, which prevents host
# sleep during long runs (a sleeping host stalls gRPC). Completed stages are
# skipped automatically, so this is safe to re-run after an interruption.
# Check progress with: jobs/status.sh
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

# One caffeinate and one resident chain at a time. -i prevents idle sleep, -m
# keeps the disk awake, -s prevents system sleep on AC power.
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
echo "Check with: jobs/status.sh"
echo "Tail with:  tail -f $LOG"
