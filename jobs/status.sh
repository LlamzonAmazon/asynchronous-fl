#!/usr/bin/env bash
# Multi-seed replicate status. Usage: jobs/status.sh
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
acc() { [ -f "$1" ] && python3 -c "import json,sys;print('%.2f%%'%json.load(open('$1'))['test_accuracy'])" 2>/dev/null || echo "-"; }

echo "===================== FL REPLICATE STATUS ====================="
date '+now: %Y-%m-%d %H:%M:%S %Z'
echo
echo "--- sleep guard ---"
if pgrep -x caffeinate >/dev/null; then echo "  caffeinate ACTIVE ($(pgrep -x caffeinate | wc -l | tr -d ' ') holders)"; else echo "  !! NO caffeinate: machine may sleep and stall gRPC"; fi
echo
echo "--- watchdog ---"
if pgrep -f "jobs/stall_watchdog.sh" >/dev/null; then
  echo "  RUNNING (pid $(pgrep -f 'jobs/stall_watchdog.sh' | head -1))"
else
  echo "  not running"
fi
WDLOG="$(ls -t jobs/logs/watchdog_*.log 2>/dev/null | head -1)"
if [ -n "$WDLOG" ]; then
  n=$(grep -ac "STALL:" "$WDLOG" 2>/dev/null); n=${n:-0}
  echo "  stall events: $n"
  [ "$n" != "0" ] && grep -aE "STALL:|relaunching|giving up" "$WDLOG" | tail -4 | sed 's/^/    /'
  echo "  last entry: $(tail -1 "$WDLOG")"
fi
echo
echo "--- chain progress ---"
BATCHLOG="$(ls -t jobs/logs/serial_*.log 2>/dev/null | head -1)"
if [ -z "$BATCHLOG" ]; then
  echo "  no serial batch log found"
else
  echo "  log: $BATCHLOG"
  grep -aE "CHAIN [0-9] of|stage .*: (starting|COMPLETE|FAILED|already complete)|chain done|aborted" "$BATCHLOG" \
    | tail -6 | sed 's/^# //; s/^\[[^]]*\] //; s/^/    /'
  last_batch="$(grep -aoE "batch [0-9]+/[0-9]+" "$BATCHLOG" | tail -1)"
  age=$(( ( $(date +%s) - $(stat -f %m "$BATCHLOG") ) / 60 ))
  echo "    progress: ${last_batch:-<no batch line yet>}   (log last written ${age}m ago)"
  if [ "$age" -ge 40 ]; then echo "    !! log is stale; watchdog should be acting"; fi
fi
echo
echo "--- results (test accuracy, round 4) ---"
printf "  %-34s %-10s %s\n" "CONFIG" "SYNC" "ASYNC K=2"
for s in 42 7 13; do
  if [ "$s" = 42 ]; then
    sp="results/sync-federated/sync_IID_4R_3C_1L/checkpoints/metrics_round_4.json"
    ap="results/async-federated/async_IID_4R_3C_1L_K2/checkpoints/metrics_round_4.json"
  else
    sp="results/sync-federated_s${s}/sync_IID_4R_3C_1L_s${s}/checkpoints/metrics_round_4.json"
    ap="results/async-federated/async_IID_4R_3C_1L_K2_s${s}/checkpoints/metrics_round_4.json"
  fi
  printf "  seed %-29s %-10s %s\n" "$s" "$(acc "$sp")" "$(acc "$ap")"
done
echo
echo "--- live workers ---"
pgrep -fl "start_server|start_client" | wc -l | xargs printf "  %s FL processes running\n"
echo "==============================================================="
