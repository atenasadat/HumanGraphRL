#!/usr/bin/env bash
# Full comparison across separate programming/writing and mixed training.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TOPGRPO_ROOT="${TOPGRPO_ROOT:-$repo_root}"
export PROFILE=main
export RUN_ROOT="${RUN_ROOT:-$TOPGRPO_ROOT/processed_data/full_training_and_benchmark_results/human_topology_full_v3}"
export PAPER_CONFIG="${PAPER_CONFIG:-$TOPGRPO_ROOT/scripts/experiment/thinkaloud_rl/paper/configs/human_topology_full_v3.json}"
exec bash "$TOPGRPO_ROOT/scripts/experiment/thinkaloud_rl/carc_paper_suite.sh" "${1:-status}"
