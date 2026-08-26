#!/usr/bin/env bash
# Run every remaining replicate chain, one at a time, in priority order.
#
# Serial by design. Each chain holds roughly 6.5 GB resident (three clients
# each load a 1.5 GB partition pickle and keep it, plus the test set and four
# Python/PyTorch runtimes). Two chains at once is about 13 GB on a 16 GB
# machine, which on 2026-08-20 drove the memory compressor to 7.4 GB, starved
# watchdogd, and panicked the kernel. One chain at a time is a hard rule now,
# not a preference.
#
# Ordering is by marginal value: seed 7 async first, because it is the single
# run that takes both arms of the headline comparison from n=1 to n=2.
#
# Completed stages are skipped on the basis of their final metrics file, so
# re-running this after an interruption resumes rather than redoes. Resume is
# stage-granular, not round-granular: an interrupted stage restarts from
# round 1.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${FL_NUM_THREADS:=3}"
: "${FL_PORT:=8081}"
export FL_NUM_THREADS FL_PORT

banner() {
  echo
  echo "############################################################"
  echo "# [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
  echo "############################################################"
}

banner "serial batch start (threads=$FL_NUM_THREADS per process, port=$FL_PORT)"

# seed 7: sync already complete at 82.44%, only async K=2 remains.
banner "CHAIN 1 of 2: seed 7, async K=2 (~5h)"
FL_SEED=7 FL_K=2 FL_STAGES=async jobs/run_seed_chain.sh
rc1=$?
banner "CHAIN 1 of 2 finished with rc=$rc1"

# seed 13: partitions already on disk, both stages still needed.
banner "CHAIN 2 of 2: seed 13, sync then async K=2 (~10h)"
FL_SEED=13 FL_K=2 FL_STAGES=both jobs/run_seed_chain.sh
rc2=$?
banner "CHAIN 2 of 2 finished with rc=$rc2"

banner "serial batch done (chain1 rc=$rc1, chain2 rc=$rc2)"
echo
echo "Results now on disk:"
find results -name "metrics_round_4.json" | sort | sed 's/^/  /'
