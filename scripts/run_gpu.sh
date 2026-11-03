#!/usr/bin/env bash
set -euo pipefail
: "${FLYBRAIN_ROOT:?}"
: "${FLYBRAIN_BASE_PYTHON:?}"
: "${FLYBRAIN_BASE_ENV:?}"
: "${FLYBRAIN_CACHE:?}"
cd "${FLYBRAIN_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mode="${FLYBRAIN_MODE:-training}"
output=output
if [[ "${mode}" == campaign-array ]]; then
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || exit 64
  output="output/task-${SLURM_ARRAY_TASK_ID}"
elif [[ "${mode}" == campaign-merge ]]; then
  output=output/merge-runtime
fi
export FLYBRAIN_JOB_OUTPUT="${output}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${PWD}/src:${PWD}"
mkdir -p "${output}"
finish() {
  rc=$?
  printf '{"exit_code":%s,"slurm_job":"%s"}\n' "${rc}" "${SLURM_JOB_ID}" > "${output}/job-exit.json"
}
trap finish EXIT
"${FLYBRAIN_BASE_PYTHON}" scripts/verify_stage.py
"${FLYBRAIN_BASE_PYTHON}" scripts/remote_runtime.py \
  "${PWD}" "${FLYBRAIN_BASE_ENV}" "${FLYBRAIN_CACHE}" > "${output}/runtime-setup.log" 2>&1
FLYBRAIN_PYTHON="$(tail -n 1 "${output}/runtime-setup.log")"
[[ -x "${FLYBRAIN_PYTHON}" ]] || { echo 'Invalid runtime receipt' >&2; exit 64; }
if [[ "${mode}" == campaign-merge ]]; then
  "${FLYBRAIN_PYTHON}" -m scripts.run_campaign --config input/train.json \
    --merge-shards output --output output/merged
  exit 0
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
egl_device="$("${FLYBRAIN_PYTHON}" -m scripts.select_remote_egl)"
[[ "${egl_device}" =~ ^[0-9]+$ ]] || { echo 'Invalid EGL selection receipt' >&2; exit 64; }
export MUJOCO_EGL_DEVICE_ID="${egl_device}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "${output}/gpu.csv"
gpu_hash="$(sha256sum "${output}/gpu.csv" | cut -d' ' -f1)"
runtime_id="$(basename "${FLYBRAIN_PYTHON%/bin/python}")"
export CUDA_CACHE_PATH="${FLYBRAIN_CACHE}/gpu/flybrain-${runtime_id}-${gpu_hash}/cuda"
mkdir -p "${CUDA_CACHE_PATH}"
"${FLYBRAIN_PYTHON}" scripts/gpu_probe.py
if [[ "${FLYBRAIN_MODE:-training}" == probe ]]; then
  exit 0
fi
if [[ "${mode}" == campaign-array ]]; then
  "${FLYBRAIN_PYTHON}" -m scripts.run_campaign --config input/train.json \
    --index "${SLURM_ARRAY_TASK_ID}" --output "${output}/campaign"
  exit 0
fi
if [[ "${FLYBRAIN_MODE:-training}" == campaign ]]; then
  "${FLYBRAIN_PYTHON}" -m scripts.run_campaign \
    --config input/train.json --output output/campaign
  exit 0
fi
"${FLYBRAIN_PYTHON}" -m flybrain_drive.cli train \
  --config input/train.json --output output/training
"${FLYBRAIN_PYTHON}" -m flybrain_drive.cli evaluate \
  --checkpoint output/training/checkpoint.pt --steps 1000 --output output/evaluation
