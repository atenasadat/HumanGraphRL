#!/usr/bin/env bash
# Run explicitly to copy pipeline code and derived priors to CARC.
set -euo pipefail
CARC_USER="${1:-saghi}"
CARC_HOST="${2:-discovery.usc.edu}"
REMOTE_DIR="/home1/${CARC_USER}/TopGRPO"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ssh "${CARC_USER}@${CARC_HOST}" "mkdir -p ${REMOTE_DIR}/scripts ${REMOTE_DIR}/processed_data"
rsync -avz --exclude '.venv*' --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/scripts/" "${CARC_USER}@${CARC_HOST}:${REMOTE_DIR}/scripts/"
rsync -avz --exclude 'rg_cache' --exclude 'experiment_results' \
    "$REPO_ROOT/processed_data/" "${CARC_USER}@${CARC_HOST}:${REMOTE_DIR}/processed_data/"
echo "Synced. On CARC, set TRAIN_VENV and run bash scripts/pilots/run_domain_pilot.sh prepare."

# Reviewed human sources are hash-verified during reference preparation.
ssh "${CARC_USER}@${CARC_HOST}" "mkdir -p ${REMOTE_DIR}/data/think-aloud"
rsync -avz "$REPO_ROOT/data/think-aloud/" "${CARC_USER}@${CARC_HOST}:${REMOTE_DIR}/data/think-aloud/"
