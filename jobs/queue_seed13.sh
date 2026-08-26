#!/usr/bin/env bash
# Wait for the seed-7 chain to finish, then run the seed-13 chain.
# Serialized deliberately: both chains hit a disk-bound partitioning phase and
# a 4-process training phase, and 16 GB does not comfortably hold two at once.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

while pgrep -f "FL_SEED=7|run_seed_chain.sh" >/dev/null 2>&1; do sleep 60; done

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] seed 7 chain finished; starting seed 13"
FL_SEED=13 FL_K=2 FL_PORT=8082 "$REPO/jobs/run_seed_chain.sh"
