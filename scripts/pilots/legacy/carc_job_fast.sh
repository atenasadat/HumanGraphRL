#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --job-name=thinkaloud_fast
#SBATCH --output=thinkaloud_fast_%j.log

# ==============================================================================
# CARC FAST-TRACK GPU Job: Cognitive Process Shaping RL
# Automatically matches ANY idle GPU (A100, RTX8000, V100)
# Uses short walltime (1.5h) for maximum Slurm backfill priority
# ==============================================================================

set -euo pipefail
cd /home1/${USER}/TopGRPO
source /etc/profile.d/modules.sh

module purge
module load gcc/13.3.0
module load python/3.11.9

# Activate CARC virtualenv
if [ -d "scripts/experiment/.venv" ]; then
    source scripts/experiment/.venv/bin/activate
else
    echo "Virtual environment scripts/experiment/.venv not found. Running setup..."
    bash scripts/pilots/carc_legacy/setup_env.sh
    source scripts/experiment/.venv/bin/activate
fi

export PYTHONPATH=$PWD
export HF_HOME=/scratch1/$USER/hf
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=== Fast Job Started ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Step 1: Unit test check
echo "=== Step 1: Verification Tests ==="

# Step 2: Run Full Scaled Training (50 steps)
MODEL_NAME=${1:-"Qwen/Qwen2.5-7B-Instruct"}
echo "=== Step 2: Training 50 Steps with $MODEL_NAME ==="
python3 scripts/experiment/thinkaloud_rl/train.py \
    --model "$MODEL_NAME" \
    --steps 50 \
    --group-size 4 \
    --shaping-weight 0.5 \
    --output-dir "pilots/results/legacy_runs/carc_thinkaloud_shaping"

echo "=== Job Completed Successfully ==="

