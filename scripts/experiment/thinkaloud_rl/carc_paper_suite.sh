#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=topgrpo_paper
#SBATCH --output=topgrpo_paper_%j.log

# One submission, one GPU, many sequential runs. Never cancels other jobs.
set -euo pipefail
cd "${TOPGRPO_ROOT:-/home1/${USER}/HumanGraphRL}"
export TOPGRPO_ROOT="$PWD"
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-/scratch1/${USER}/hf}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PROFILE="${PROFILE:-pilot}"
if [[ "$PROFILE" == pilot ]]; then
    default_run_root="$PWD/pilots/results/human_topology_pilot_v4"
else
    default_run_root="$PWD/processed_data/full_training_and_benchmark_results/paper_${PROFILE}"
fi
export RUN_ROOT="${RUN_ROOT:-$default_run_root}"
export PAPER_DATA="${PAPER_DATA:-/scratch1/${USER}/humangraphrl_public_data_v2}"
export PAPER_VENV="${PAPER_VENV:-$PWD/scripts/experiment/.venv-paper}"
action="${1:-submit}"

if [[ "$action" == prepare ]]; then
    # Reuse the existing working CUDA build, without installing over the training environment.
    if [[ ! -x "$PAPER_VENV/bin/python" ]]; then
        "${TRAIN_VENV:-scripts/experiment/.venv}/bin/python" -m venv --system-site-packages "$PAPER_VENV"
    fi
    # Repair partially created environments on repeated preparation attempts.
    train_site=$("${TRAIN_VENV:-scripts/experiment/.venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    paper_site=$("$PAPER_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # A venv does not normally inherit another venv's packages. Explicitly
    # expose the working CUDA environment after this environment's own packages.
    printf '%s\n' "$train_site" > "$paper_site/topgrpo_training_environment.pth"
    if ! "$PAPER_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        "$PAPER_VENV/bin/python" -m ensurepip --upgrade
    fi
    "$PAPER_VENV/bin/python" -m pip install -r scripts/experiment/thinkaloud_rl/requirements-paper.txt
    "$PAPER_VENV/bin/python" -m pip freeze > "$PAPER_VENV/requirements-resolved.txt"
    init_args=(--root "$RUN_ROOT" --data-root "$PAPER_DATA" --profile "$PROFILE")
    if [[ -n "${PAPER_CONFIG:-}" ]]; then init_args+=(--config "$PAPER_CONFIG"); fi
    if [[ ! -f "$RUN_ROOT/plan.json" ]]; then
        "$PAPER_VENV/bin/python" -m scripts.experiment.thinkaloud_rl.paper.suite init "${init_args[@]}"
    fi
    "$PAPER_VENV/bin/python" -m scripts.experiment.thinkaloud_rl.paper.suite prepare --root "$RUN_ROOT"
    echo "Prepared. Submit ONE job with: RUN_ROOT=$RUN_ROOT bash scripts/experiment/thinkaloud_rl/carc_paper_suite.sh submit"
    exit 0
fi

if [[ ! -x "$PAPER_VENV/bin/python" || ! -f "$RUN_ROOT/prepared.json" ]]; then
    echo "Run this script with 'prepare' on the login node first." >&2
    exit 1
fi

if [[ "$action" == status ]]; then
    exec "$PAPER_VENV/bin/python" -m scripts.experiment.thinkaloud_rl.paper.suite status --root "$RUN_ROOT"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ "$action" == submit ]] || { echo "Expected prepare, submit, or status" >&2; exit 2; }
    # Reject stale code/plans and missing full-run pilot prerequisites before queueing.
    "$PAPER_VENV/bin/python" -m scripts.experiment.thinkaloud_rl.paper.suite gate --root "$RUN_ROOT"
    # Each resubmission resumes the same root. No job arrays or automatic queue flooding.
    exec sbatch --export=ALL "$PWD/scripts/experiment/thinkaloud_rl/carc_paper_suite.sh" worker
fi

source "$PAPER_VENV/bin/activate"
python - <<'PY'
import torch
import bitsandbytes as bnb
if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
layer=bnb.nn.Linear4bit(64,64,bias=False,compute_dtype=torch.float16,quant_type='nf4').to('cuda')
x=torch.randn(2,64,device='cuda',dtype=torch.float16,requires_grad=True)
layer(x).float().square().mean().backward()
if not torch.isfinite(x.grad).all(): raise RuntimeError('Nonfinite NF4 preflight gradient')
print('NF4 GPU forward/backward preflight passed')
q=torch.randn(1,2,16,64,device='cuda',dtype=torch.float16,requires_grad=True)
torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True).float().square().mean().backward()
if not torch.isfinite(q.grad).all(): raise RuntimeError('Nonfinite SDPA preflight gradient')
print('SDPA GPU forward/backward preflight passed')
PY

worker_pid=""
forward_stop() {
    if [[ -n "$worker_pid" ]]; then kill -USR1 "$worker_pid" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM
python -m scripts.experiment.thinkaloud_rl.paper.suite run --root "$RUN_ROOT" \
    --max-seconds "${PAPER_MAX_SECONDS:-41400}" &
worker_pid=$!
set +e
wait "$worker_pid"
rc=$?
# A signal may interrupt shell wait before the child finishes its checkpoint.
while kill -0 "$worker_pid" 2>/dev/null; do wait "$worker_pid"; rc=$?; done
if (( rc >= 128 )); then wait "$worker_pid"; rc=$?; fi
set -e
if [[ "$rc" == 75 ]]; then
    echo "Checkpointed. Resume with the SAME RUN_ROOT and one new submission."
    exit 0
fi
exit "$rc"
