#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --job-name=thinkaloud_rl
#SBATCH --output=thinkaloud_rl_%j.log

# ==============================================================================
# CARC GPU Job: Cognitive Process Shaping RL Training
# Runs on USC Discovery Cluster with 1 A100 GPU
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

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Verify unit tests first
echo "=== Step 1: Running Unit Tests ==="

# 1. Condition A: Baseline (Outcome-only RL, shaping weight = 0.0)
echo "=== Step 2: Training Condition A (Baseline, Weight = 0.0) ==="
python3 scripts/pilots/legacy/run_pilot_shaping_comparison.py

# 2. Condition B: Full Scaled Training with Qwen2.5-7B or Qwen3-8B
MODEL_NAME=${1:-"Qwen/Qwen2.5-7B-Instruct"}
echo "=== Step 3: Running Scaled Training with $MODEL_NAME ==="
python3 scripts/experiment/thinkaloud_rl/train.py \
    --model "$MODEL_NAME" \
    --steps 50 \
    --group-size 4 \
    --shaping-weight 0.5 \
    --output-dir "pilots/results/legacy_runs/carc_thinkaloud_shaping"

echo "=== Job Completed Successfully ==="

