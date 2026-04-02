#!/bin/bash
#
# Submit centralized + sync FL + async FL jobs with dependency chaining.
# Async waits for sync to finish (needs partition .pkl files).
#
# Usage (from repo root):
#   bash jobs/submit_all.sh
#
# To submit only sync + async (skip centralized):
#   bash jobs/submit_all.sh --skip-centralized

set -e

SKIP_CENTRALIZED=false
for arg in "$@"; do
    case $arg in
        --skip-centralized) SKIP_CENTRALIZED=true ;;
    esac
done

echo "============================================"
echo "Submitting thesis experiment jobs"
echo "============================================"

# 1. Centralized (independent)
if [ "$SKIP_CENTRALIZED" = false ]; then
    CENT_JOB=$(sbatch --parsable jobs/job_centralized.sh)
    echo "Centralized submitted: job $CENT_JOB"
fi

# 2. Sync FL (must run before async to generate partition data)
SYNC_JOB=$(sbatch --parsable jobs/job_sync_fl.sh)
echo "Sync FL submitted:     job $SYNC_JOB"

# 3. Async FL (depends on sync finishing successfully)
ASYNC_JOB=$(sbatch --parsable --dependency=afterok:$SYNC_JOB jobs/job_async_fl.sh)
echo "Async FL submitted:    job $ASYNC_JOB (depends on $SYNC_JOB)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel -u \$USER"
echo "============================================"
