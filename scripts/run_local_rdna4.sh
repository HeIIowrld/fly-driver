#!/usr/bin/env bash
# Local feedback complements, and does not modify, the remote Slurm campaign.
set -euo pipefail
: "${MECAR_RDNA4_PYTHON:?Set this to your verified ROCm Python executable}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export LP_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
config="${1:-configs/campaign-rdna4-screening.json}"
output="${2:?Pass a new runs/ output directory}"
"$MECAR_RDNA4_PYTHON" -m scripts.run_campaign --config "$config" --output "$output"
"$MECAR_RDNA4_PYTHON" -m scripts.select_local_candidates \
  --summary "$output/summary.json" --output "$output/selection.json"
