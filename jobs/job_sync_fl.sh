#!/bin/bash
#SBATCH --account=def-zublMDM           # Replace with your professor's Alliance account
#SBATCH --time=02:00:00                 # Wall-clock limit
#SBATCH --cpus-per-task=8               # CPU workers (server + 3 clients)
#SBATCH --mem=32G                       # RAM for multiple processes
#SBATCH --gres=gpu:a100:1              # One A100 GPU on Narval
#SBATCH --job-name=sync_fl
#SBATCH --output=jobs/logs/%x_%j.out    # Stdout/stderr → jobs/logs/sync_fl_<jobid>.out

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python federated/synchronous/run_fl.py

echo "Job finished: $(date)"
