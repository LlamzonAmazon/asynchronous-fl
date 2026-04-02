#!/bin/bash
#SBATCH --account=def-zublMDM          # Replace with your professor's Alliance account
#SBATCH --time=01:00:00                 # Wall-clock limit (centralized is fastest)
#SBATCH --cpus-per-task=4               # CPU workers for DataLoader
#SBATCH --mem=16G                       # RAM for data loading + model
#SBATCH --gres=gpu:a100:1              # One A100 GPU on Narval
#SBATCH --job-name=centralized
#SBATCH --output=jobs/logs/%x_%j.out    # Stdout/stderr → jobs/logs/centralized_<jobid>.out

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python centralized/train.py

echo "Job finished: $(date)"
