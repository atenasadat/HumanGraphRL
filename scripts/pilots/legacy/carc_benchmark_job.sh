#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_2026_benchmarks
#SBATCH --output=eval_2026_benchmarks_%j.log

set -euo pipefail

cd "${TOPGRPO_ROOT:-/home1/${USER}/TopGRPO}"
source "${TRAIN_VENV:-scripts/experiment/.venv}/bin/activate"

export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-/scratch1/${USER}/benchmarks}"
export OUTPUT_DIR="${OUTPUT_DIR:-pilots/results/legacy_runs/benchmark_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================================================="
echo "CARC 2024-2026 Modern Gold-Standard Benchmark Evaluation"
echo "Host: $(hostname)"
echo "Benchmark dir on scratch: ${BENCHMARK_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================================================="

# 1. Download official modern benchmarks directly onto CARC scratch if missing
bash scripts/experiment/thinkaloud_rl/download_benchmarks.sh "${BENCHMARK_DIR}"

# 2. Run LiveCodeBench (LCB 2024-2026: Contamination-free competitive coding)
echo "==> Running LiveCodeBench (LCB) evaluation..."
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --adapter "${LORA_ADAPTER:-}" \
    --benchmark livecodebench \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --num-samples 1 \
    --limit 100 \
    --output-file "${OUTPUT_DIR}/livecodebench_results.json"

# 3. Run MATH-500 (DeepSeek-R1 / OpenAI o1 Standard)
echo "==> Running MATH-500 mathematical reasoning evaluation..."
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --adapter "${LORA_ADAPTER:-}" \
    --benchmark math500 \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --num-samples 1 \
    --limit 100 \
    --output-file "${OUTPUT_DIR}/math500_results.json"

# 4. Run AIME 2024 (Olympiad-level multi-step mathematical reasoning)
echo "==> Running AIME 2024 evaluation..."
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --adapter "${LORA_ADAPTER:-}" \
    --benchmark aime \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --num-samples 1 \
    --output-file "${OUTPUT_DIR}/aime2024_results.json"

# 5. Run WildBench (Allen AI 2024: Scholarly & Complex Constraint Writing)
echo "==> Running WildBench writing evaluation..."
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --adapter "${LORA_ADAPTER:-}" \
    --benchmark wildbench \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --output-file "${OUTPUT_DIR}/wildbench_results.json"

echo "=============================================================================="
echo "[✓] All modern 2024-2026 benchmark evaluations completed successfully!"
echo "Results stored in: ${OUTPUT_DIR}"
echo "=============================================================================="
