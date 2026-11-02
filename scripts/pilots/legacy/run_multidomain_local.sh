#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-scripts/experiment/.venv/bin/python}
exec "$PYTHON_BIN" -m scripts.pilots.legacy.run_multidomain_suite "$@"
