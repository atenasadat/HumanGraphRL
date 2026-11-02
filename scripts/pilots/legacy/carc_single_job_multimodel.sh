#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=multimodel_eval
#SBATCH --output=multimodel_eval_%j.log

set -euo pipefail

cd "${TOPGRPO_ROOT:-/home1/${USER}/TopGRPO}"
source "${TRAIN_VENV:-scripts/experiment/.venv}/bin/activate"

export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-/scratch1/${USER}/benchmarks}"
export OUTPUT_DIR="${OUTPUT_DIR:-pilots/results/legacy_runs/multimodel_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================================================="
echo "CARC Single-Job Multi-Model Comparative Evaluation"
echo "Host: $(hostname)"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================================================="

# 1. Ensure modern 2024-2026 benchmarks are present on scratch
bash scripts/experiment/thinkaloud_rl/download_benchmarks.sh "${BENCHMARK_DIR}"

# 2. Run multi-model evaluation sequentially in ONE job allocation:
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# - Qwen/Qwen2.5-Coder-7B-Instruct
# - Qwen/Qwen2.5-7B-Instruct
# - meta-llama/Llama-3.1-8B-Instruct
python -m scripts.pilots.legacy.run_multimodel_suite \
    --models deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
             Qwen/Qwen2.5-Coder-7B-Instruct \
             Qwen/Qwen2.5-7B-Instruct \
    --benchmarks livecodebench math500 aime wildbench \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --limit 50 \
    --max-tokens 1536 \
    --load-in-4bit

echo "=============================================================================="
echo "[✓] Multi-model evaluation finished cleanly!"
echo "Check summary: ${OUTPUT_DIR}/multimodel_comparative_summary.json"
echo "=============================================================================="

