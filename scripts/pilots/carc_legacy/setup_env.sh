#!/bin/bash
# One-time setup on the CARC login node:   cd /home1/saghi/TopGRPO && bash scripts/pilots/carc_legacy/setup_env.sh
# Creates a Python virtual env in scripts/experiment/.venv (CARC python module) and pre-downloads every model the jobs need.
set -euo pipefail
cd "$(dirname "$0")/.."
source /etc/profile.d/modules.sh

module purge
module load gcc/13.3.0
module load python/3.11.9

python3 -m venv scripts/experiment/.venv
source scripts/experiment/.venv/bin/activate
pip install --upgrade pip
pip install "torch==2.8.0" "transformers>=4.51,<4.56" "peft>=0.15,<0.18" "accelerate>=1.0" \
    pyyaml pandas pyarrow matplotlib numpy huggingface_hub

export HF_HOME=/scratch1/$USER/hf
mkdir -p "$HF_HOME"
for m in Qwen/Qwen3-1.7B sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
    huggingface-cli download "$m" --exclude "*.onnx" "*.h5" "*.ot" "*.msgpack" "openvino/*" "onnx/*"
done
python3 -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
echo "setup done"
