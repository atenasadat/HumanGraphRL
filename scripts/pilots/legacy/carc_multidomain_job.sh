#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-26%2
#SBATCH --job-name=cognitive_domains
#SBATCH --output=cognitive_domains_%A_%a.log
set -euo pipefail
cd "${TOPGRPO_ROOT:-/home1/${USER}/TopGRPO}"
source "${TRAIN_VENV:-scripts/experiment/.venv}/bin/activate"
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep login-node checks and GPU workers from spawning host-sized BLAS pools.
# Override inherited values (often 64) before any Python imports.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Run with bash on the login node to replace the previous experiment arrays.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    python -c 'import torch, transformers, peft, bitsandbytes'
    python -m scripts.experiment.thinkaloud_rl.train --help | grep -q -- '--load-in-4bit'
    script="$PWD/scripts/pilots/legacy/carc_multidomain_job.sh"
    nodes=(b11-12 d13-08 d14-06 d14-07 d14-08 d14-16 e21-12)
    types=(a40 v100 v100 v100 v100 v100 p100)
    # Capture only this user's named experiment jobs BEFORE submitting replacements.
    old_jobs=$(squeue -h -u "$USER" -n cognitive_domains -o '%i' |
        sed 's/_.*//' | sort -u)
    export RUN_ROOT="${RUN_ROOT:-pilots/results/legacy_runs/carc_multidomain_$(date +%Y%m%d_%H%M%S)_$$}"
    mkdir -p "$RUN_ROOT"
    new_jobs=()
    cleanup() {
        for job in "${new_jobs[@]}"; do scancel "$job" || true; done
    }
    trap cleanup ERR
    # Submit held replacements first. If submission fails, old jobs remain intact.
    for n in "${!nodes[@]}"; do
        indices=""
        for ((i=n; i<27; i+=7)); do indices="${indices:+${indices},}${i}"; done
        job=$(sbatch --hold --parsable --nodelist="${nodes[$n]}" \
            --gpus-per-task="${types[$n]}:1" --array="${indices}%2" \
            --export=ALL,RUN_ROOT="$RUN_ROOT" "$script")
        job=${job%%;*}
        new_jobs+=("$job")
        printf '%s %s indices=%s job=%s\n' "${nodes[$n]}" "${types[$n]}" "$indices" "$job" |
            tee -a "$RUN_ROOT/submissions.txt"
    done
    for job in $old_jobs; do
        [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid old job ID: $job" >&2; exit 1; }
        scancel "$job"
    done
    # Preserve accepted replacement IDs if a later release needs manual retry.
    trap - ERR
    for job in "${new_jobs[@]}"; do scontrol release "$job"; done
    echo "Submitted 27 pilots/results/legacy_runs across seven nodes, at most 14 simultaneous GPUs."
    echo "Results: $RUN_ROOT"
    exit 0
fi

# All GPU types use the SAME NF4 base model, avoiding full-precision 7B on P100.
python - <<'PY'
import torch
import bitsandbytes as bnb
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable')
print('GPU:', torch.cuda.get_device_name(), 'Torch CUDA:', torch.version.cuda)
layer = bnb.nn.Linear4bit(16, 16, bias=False, compute_dtype=torch.float16,
                         quant_type='nf4').to('cuda')
x = torch.randn(2, 16, device='cuda', dtype=torch.float16, requires_grad=True)
layer(x).float().square().mean().backward()
print('NF4 forward/backward check passed')
PY
python -m scripts.pilots.legacy.run_multidomain_suite \
  --model "${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" \
  --output-root "${RUN_ROOT:-pilots/results/legacy_runs/carc_multidomain_${SLURM_ARRAY_JOB_ID}}" \
  --steps "${TRAIN_STEPS:-50}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --load-in-4bit --job-index "${SLURM_ARRAY_TASK_ID}" --execute
