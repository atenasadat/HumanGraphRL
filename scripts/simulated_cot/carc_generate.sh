#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --job-name=simulated_cot
#SBATCH --output=simulated_cot_%A_%a.log
# Usage (from the repository root on a CARC login node):
#   bash scripts/simulated_cot/carc_generate.sh setup    # venv + pinned model downloads (~45 GB in $HF_HOME)
#   bash scripts/simulated_cot/carc_generate.sh submit   # one A100 job per model
# Resubmitting resumes: finished samples are skipped.
set -euo pipefail
cd "${TOPGRPO_ROOT:-/home1/${USER}/HumanGraphRL}"
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# vLLM with its torch/CUDA wheels is ~10 GB; keep it on scratch, not the home quota.
VENV="${SIMCOT_VENV:-/scratch1/${USER}/venvs/simulated_cot}"
MODELS=(R1-Distill-Qwen-7B Qwen3-8B gpt-oss-20b)
action="${1:-worker}"

if [[ "$action" == setup ]]; then
    for f in data/writing_instructions/manifest.json data/programming_instructions/manifest.json; do
        [[ -f "$f" ]] || { echo "Missing $f: build instructions locally and copy data/ first" >&2; exit 1; }
    done
    [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r scripts/simulated_cot/requirements.txt
    "$VENV/bin/python" -m pip freeze > "$VENV/requirements-resolved.txt"
    "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
from scripts.simulated_cot.generate import MODELS
# gpt-oss also ships original/ and metal/ copies of its weights (27.6 GB) that vLLM never reads.
for name, spec in MODELS.items():
    print(name, snapshot_download(spec['repo'], revision=spec['revision'],
                                  ignore_patterns=['original/*', 'metal/*', 'figures/*']))
PY
    exit 0
fi
if [[ "$action" == submit ]]; then
    [[ -x "$VENV/bin/python" ]] || { echo "Run setup first" >&2; exit 1; }
    exec sbatch --export=ALL "$PWD/scripts/simulated_cot/carc_generate.sh" worker
fi

model="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HF_HUB_OFFLINE=1
exec "$VENV/bin/python" -m scripts.simulated_cot.generate --model "$model"
