#!/usr/bin/env bash
# Optional hardware-camera path; run in the verified WSL ROCm environment.
set -euo pipefail
: "${MECAR_RDNA4_PYTHON:?Set this to your verified ROCm Python executable}"
cd "$(dirname "$0")/.."
config="${1:?Pass an experiment config}"
output="${2:?Pass a new runs/ output directory}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MIOPEN_FIND_MODE=FAST
export MUJOCO_GL=glfw GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The gate includes actual four-worker GPU PPO and preserves a bounded failure log.
"$MECAR_RDNA4_PYTHON" -m scripts.benchmark_acceleration \
  --config "$config" --output "${output}-renderer-gate" --timeout-seconds 180
"$MECAR_RDNA4_PYTHON" - "$output-renderer-gate/result.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
case = next(row for row in result["cases"] if row["renderer"] == "d3d12")
name = case.get("profile", {}).get("graphics", {}).get("renderer", "")
if case["status"] != "completed" or "D3D12 (AMD Radeon RX 9070 XT)" not in name:
    raise SystemExit("Hardware camera / GPU PPO gate failed; review its logs")
if not result["fixed_command_equivalence"].get("physics_equal", False):
    raise SystemExit("Renderer physics equivalence gate failed")
print("Verified GPU camera renderer:", name)
PY
bash scripts/run_local_rdna4.sh "$config" "$output"
