#!/usr/bin/env bash
# Detect a stalled FL batch and restart it.
#
#   jobs/stall_watchdog.sh <serial-log-path>
#
# A chain can deadlock in Flower's gRPC layer (observed at round 2, 0% CPU for
# many hours) without exiting. This detects that case.
#
# Staleness is judged by the mtime of the batch log. Every client flushes a
# progress line every 20 batches and three clients train concurrently, so a
# healthy run writes to the log far more often than the stall threshold.
# Log mtime is used because inferring health from the process tree and CPU
# percentages can misidentify the chain root (for example, caffeinate).
#
# The restart is safe to repeat: completed stages are skipped on the basis of
# their final metrics file, and an incomplete stage directory is archived
# rather than resumed into.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOG="${1:?usage: stall_watchdog.sh <serial-log-path>}"
: "${STALL_MIN:=40}"
: "${MAX_RESTARTS:=2}"
: "${POLL_SEC:=300}"

WLOG="jobs/logs/watchdog_$(date '+%m%d_%H%M').log"
wlog() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$WLOG"; }

wlog "watching $LOG (stall=${STALL_MIN}m, poll=${POLL_SEC}s, max restarts=$MAX_RESTARTS)"

restarts=0

kill_batch() {
  wlog "killing batch process tree"
  pkill -f "caffeinate -ims jobs/run_chains_serial.sh" 2>/dev/null
  pkill -f "jobs/run_chains_serial.sh"                 2>/dev/null
  pkill -f "jobs/run_seed_chain.sh"                    2>/dev/null
  pkill -f "federated/(synchronous|asynchronous)/run_fl.py" 2>/dev/null
  pkill -f "start_server.py"                           2>/dev/null
  pkill -f "start_client.py"                           2>/dev/null
  sleep 20
  pkill -9 -f "start_server.py" 2>/dev/null
  pkill -9 -f "start_client.py" 2>/dev/null
  sleep 5
}

while true; do
  sleep "$POLL_SEC"

  if ! pgrep -f "jobs/run_chains_serial.sh" >/dev/null 2>&1; then
    wlog "serial batch is no longer running; watchdog exiting"
    exit 0
  fi

  if [ ! -f "$LOG" ]; then
    wlog "log $LOG missing; skipping this poll"
    continue
  fi

  age_min=$(( ( $(date +%s) - $(stat -f %m "$LOG") ) / 60 ))

  if [ "$age_min" -lt "$STALL_MIN" ]; then
    continue
  fi

  wlog "STALL: $LOG has not been written for ${age_min}m (threshold ${STALL_MIN}m)"

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    wlog "restart budget exhausted ($restarts); killing batch and giving up"
    kill_batch
    exit 1
  fi

  kill_batch
  restarts=$(( restarts + 1 ))
  wlog "relaunching batch (restart $restarts of $MAX_RESTARTS)"
  jobs/launch_batch.sh >> "$WLOG" 2>&1

  # launch_batch.sh writes a fresh timestamped log; follow the new one.
  sleep 10
  newest="$(ls -t jobs/logs/serial_*.log 2>/dev/null | head -1)"
  if [ -n "$newest" ] && [ "$newest" != "$LOG" ]; then
    LOG="$newest"
    wlog "now watching $LOG"
  fi
done
