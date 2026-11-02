#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=head_to_head_eval
#SBATCH --output=head_to_head_%j.log

set -euo pipefail

cd "${TOPGRPO_ROOT:-/home1/${USER}/TopGRPO}"
source "${TRAIN_VENV:-scripts/experiment/.venv}/bin/activate"

export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-/scratch1/${USER}/benchmarks}"
export OUTPUT_DIR="${OUTPUT_DIR:-pilots/results/legacy_runs/head_to_head_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
BASE_DIR="pilots/results/legacy_runs/carc_multidomain_20261027_153910_2209184"

BASELINE_ADAPTER="${BASE_DIR}/mixed_baseline_seed17/adapter"
OUR_COGSHAPE_ADAPTER="${BASE_DIR}/mixed_unified_seed17/adapter"

echo "=============================================================================="
echo "CARC Head-to-Head Evaluation: Baseline vs. Our CogShape Model"
echo "Base Model: ${BASE_MODEL}"
echo "Unshaped Baseline Adapter: ${BASELINE_ADAPTER}"
echo "Our CogShape Model Adapter: ${OUR_COGSHAPE_ADAPTER}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=============================================================================="

# 1. Ensure benchmarks exist on scratch
bash scripts/experiment/thinkaloud_rl/download_benchmarks.sh "${BENCHMARK_DIR}"

# ==============================================================================
# EVALUATION 1: OUR COGSHAPE MODEL (MIXED UNIFIED SEED 17)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "--> [1/4] Evaluating OUR COGSHAPE MODEL on LiveCodeBench..."
echo "=============================================================================="
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL}" \
    --adapter "${OUR_COGSHAPE_ADAPTER}" \
    --benchmark livecodebench \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --limit 50 \
    --output-file "${OUTPUT_DIR}/our_cogshape_livecodebench.json"

echo ""
echo "=============================================================================="
echo "--> [2/4] Evaluating OUR COGSHAPE MODEL on MATH-500..."
echo "=============================================================================="
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL}" \
    --adapter "${OUR_COGSHAPE_ADAPTER}" \
    --benchmark math500 \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --limit 50 \
    --output-file "${OUTPUT_DIR}/our_cogshape_math500.json"

# ==============================================================================
# EVALUATION 2: UNSHAPED BASELINE (MIXED BASELINE SEED 17)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "--> [3/4] Evaluating UNSHAPED BASELINE on LiveCodeBench..."
echo "=============================================================================="
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL}" \
    --adapter "${BASELINE_ADAPTER}" \
    --benchmark livecodebench \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --limit 50 \
    --output-file "${OUTPUT_DIR}/baseline_livecodebench.json"

echo ""
echo "=============================================================================="
echo "--> [4/4] Evaluating UNSHAPED BASELINE on MATH-500..."
echo "=============================================================================="
python -m scripts.experiment.thinkaloud_rl.evaluate_benchmark \
    --base-model "${BASE_MODEL}" \
    --adapter "${BASELINE_ADAPTER}" \
    --benchmark math500 \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --load-in-4bit \
    --limit 50 \
    --output-file "${OUTPUT_DIR}/baseline_math500.json"

# ==============================================================================
# SUMMARIZE HEAD-TO-HEAD COMPARISON
# ==============================================================================
echo ""
echo "=============================================================================="
echo "--> Compiling Head-to-Head Comparison Summary..."
echo "=============================================================================="
python - << 'EOF'
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUTPUT_DIR", "."))

def get_stats(filename):
    p = out_dir / filename
    if not p.exists():
        return {"accuracy": "N/A", "entropy": "N/A"}
    d = json.loads(p.read_text())
    acc = d.get("accuracy_or_pass_rate", 0) * 100
    cog = d.get("cognitive_metrics", {})
    ent = cog.get("move_entropy_bits", "N/A")
    return {"accuracy": f"{acc:.2f}%", "entropy": ent}

summary = {
    "LiveCodeBench": {
        "Unshaped Baseline": get_stats("baseline_livecodebench.json"),
        "Our CogShape Model": get_stats("our_cogshape_livecodebench.json")
    },
    "MATH-500": {
        "Unshaped Baseline": get_stats("baseline_math500.json"),
        "Our CogShape Model": get_stats("our_cogshape_math500.json")
    }
}

summary_file = out_dir / "head_to_head_summary.json"
summary_file.write_text(json.dumps(summary, indent=2))
print(f"[✓] Summary written to: {summary_file}")
print(json.dumps(summary, indent=2))
EOF

echo "=============================================================================="
echo "[✓] Head-to-Head Evaluation Completed!"
echo "Results saved in: ${OUTPUT_DIR}"
echo "=============================================================================="

