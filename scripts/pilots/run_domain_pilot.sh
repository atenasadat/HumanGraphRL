#!/usr/bin/env bash
# Prepare/submit/status for the separate-context and mixed-context pilot.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TOPGRPO_ROOT="${TOPGRPO_ROOT:-$repo_root}"
export PROFILE=pilot
export TRAIN_VENV="${TRAIN_VENV:-$TOPGRPO_ROOT/humangraphrl}"
export PAPER_VENV="${PAPER_VENV:-$TRAIN_VENV}"
export RUN_ROOT="${RUN_ROOT:-$TOPGRPO_ROOT/pilots/results/human_topology_pilot_v3_calibration_audit}"
export PAPER_CONFIG="${PAPER_CONFIG:-$TOPGRPO_ROOT/scripts/experiment/thinkaloud_rl/paper/configs/human_topology_pilot_v3.json}"
exec bash "$TOPGRPO_ROOT/scripts/experiment/thinkaloud_rl/carc_paper_suite.sh" "${1:-status}"
