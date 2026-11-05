#!/usr/bin/env bash
# Run on the GPU VM. "check" also works in Cloud Shell without CUDA packages.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PWD"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VENV="${SIMCOT_VENV:-$PWD/.venv-cot}"
action="${1:-check}"
shift || true

if [[ "$action" == check ]]; then
    exec python3 -m scripts.simulated_cot.generate --check-inputs
fi
case "$action" in drivers|setup|smoke|run) ;; *) echo 'Usage: bash scripts/simulated_cot/gcp_generate.sh check|drivers|setup|smoke|run [model ...]' >&2; exit 2 ;; esac

# Prevent installing CUDA dependencies in Cloud Shell or on the researcher's Mac.
[[ "$(uname -s)" == Linux ]] || { echo 'Run this action on the Google GPU VM.' >&2; exit 1; }
gpu_found=false
for vendor in /sys/bus/pci/devices/*/vendor; do
    if [[ "$(cat "$vendor")" == 0x10de ]]; then gpu_found=true; break; fi
done
[[ "$gpu_found" == true ]] || { echo 'No NVIDIA PCI device: connect to the GPU VM first, not Cloud Shell.' >&2; exit 1; }

if [[ "$action" == drivers ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
        exec nvidia-smi
    fi
    echo 'Google driver installer may reboot this VM. Reconnect and repeat drivers if interrupted.'
    sudo apt-get update
    sudo apt-get install -y python3 curl
    installer=$(mktemp /var/tmp/humangraphrl-driver.XXXXXX)
    trap 'rm -f "$installer"' EXIT
    curl -fSsL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o "$installer"
    sudo python3 "$installer" install_driver --installation-mode=repo --installation-branch=prod
    echo 'If nvidia-smi fails, reboot the VM, reconnect, and repeat drivers.'
    nvidia-smi
    exit 0
fi

nvidia-smi >/dev/null || { echo 'Run the drivers action first.' >&2; exit 1; }
python3 -m scripts.simulated_cot.generate --check-inputs
if [[ "$action" == setup ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-venv tmux
    [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r scripts/simulated_cot/requirements.txt
    "$VENV/bin/python" -m pip freeze > "$VENV/requirements-resolved.txt"
    "$VENV/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable after installation')
if torch.cuda.get_device_capability()[0] < 8:
    raise RuntimeError('BF16/MXFP4 requires an Ampere or newer GPU')
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print('BF16 CUDA check passed:', torch.cuda.get_device_name())
from huggingface_hub import snapshot_download
from scripts.simulated_cot.generate import MODELS
for name, spec in MODELS.items():
    print('Downloading', name, flush=True)
    snapshot_download(spec['repo'], revision=spec['revision'],
                      ignore_patterns=['original/*', 'metal/*', 'figures/*'])
PY
    exit 0
fi

[[ -x "$VENV/bin/python" ]] || { echo 'Run setup first.' >&2; exit 1; }
models=(R1-Distill-Qwen-7B Qwen3-8B gpt-oss-20b)
if (( $# )); then models=("$@"); fi
mkdir -p processed_data/cloud_logs
export HF_HUB_OFFLINE=1
for model in "${models[@]}"; do
    case "$model" in R1-Distill-Qwen-7B|Qwen3-8B|gpt-oss-20b) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
    args=(--model "$model" --profile l4 --batch 1)
    if [[ "$action" == smoke ]]; then args+=(--smoke-test); fi
    "$VENV/bin/python" -m scripts.simulated_cot.generate "${args[@]}" \
        2>&1 | tee -a "processed_data/cloud_logs/${model}_${action}.log"
done
