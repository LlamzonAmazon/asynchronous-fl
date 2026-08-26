#!/usr/bin/env bash
# Detach the stall watchdog against a batch log.
#
#   jobs/start_watchdog.sh <serial-log-path>
#
# Exists as its own file because backgrounding from an interactive or
# tool-driven shell does not reliably survive that shell exiting, even with
# nohup and disown. Launching from inside a script does.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

LOG="${1:?usage: start_watchdog.sh <serial-log-path>}"
: "${STALL_MIN:=40}"
: "${MAX_RESTARTS:=2}"
: "${POLL_SEC:=300}"
export STALL_MIN MAX_RESTARTS POLL_SEC

if pgrep -f "jobs/stall_watchdog.sh" >/dev/null 2>&1; then
  echo "watchdog already running."; exit 0
fi

nohup jobs/stall_watchdog.sh "$LOG" >/dev/null 2>&1 &
disown
sleep 2

if pgrep -f "jobs/stall_watchdog.sh" >/dev/null 2>&1; then
  echo "watchdog started (stall=${STALL_MIN}m, poll=${POLL_SEC}s, max restarts=${MAX_RESTARTS})"
  echo "watching: $LOG"
else
  echo "WARNING: watchdog failed to start."; exit 1
fi
