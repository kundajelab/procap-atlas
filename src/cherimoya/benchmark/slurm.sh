#!/usr/bin/env bash
#SBATCH --job-name=cherimoya_benchmark
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=12:00:00
#SBATCH --output=logs/cherimoya_benchmark_%j.out
#SBATCH --error=logs/cherimoya_benchmark_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/users/${USER}/numba_cache}"
CHERIMOYA_VENV_DIR="${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"
FORCE="${FORCE:-0}"

if [[ ! -d "${REPO_ROOT}/models/cherimoya" ]]; then
    echo "Could not find ${REPO_ROOT}/models/cherimoya." >&2
    echo "Submit this script from the procap-atlas repo root, or set SLURM_SUBMIT_DIR to the repo root." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
export NUMBA_CACHE_DIR

mkdir -p "${NUMBA_CACHE_DIR}"

# Native (non-Apptainer) Cherimoya environment: see
# src/cherimoya/sherlock_native/README.md. The NGC-based Apptainer image
# cannot run on Sherlock's GPU driver.
ml load math
ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314
source "${CHERIMOYA_VENV_DIR}/bin/activate"

if [[ $# -gt 0 ]]; then
    model_names=("$@")
else
    model_names=()
    shopt -s nullglob
    for model_dir in "${REPO_ROOT}"/models/cherimoya/*; do
        [[ -d "${model_dir}" ]] || continue
        model_names+=("$(basename "${model_dir}")")
    done
    shopt -u nullglob
fi

if [[ ${#model_names[@]} -eq 0 ]]; then
    echo "No Cherimoya model directories found under ${REPO_ROOT}/models/cherimoya" >&2
    exit 1
fi

cd "${REPO_ROOT}"
nvidia-smi -L

for model_name in "${model_names[@]}"; do
    exp_id="${model_name%%_*}"
    model_dir="${REPO_ROOT}/models/cherimoya/${model_name}"
    metrics_path="${REPO_ROOT}/performance_metrics/cherimoya/${model_name}.json"

    if [[ ! -d "${model_dir}" ]]; then
        echo "Skipping ${model_name}: model directory not found at ${model_dir}" >&2
        continue
    fi

    if [[ "${FORCE}" != "1" && -f "${metrics_path}" ]]; then
        echo "Skipping ${model_name}: benchmark already exists at ${metrics_path}"
        continue
    fi

    echo "Benchmarking ${model_name} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    python3 "${REPO_ROOT}/src/cherimoya/benchmark/benchmark_cherimoya.py" \
        -e "${exp_id}" \
        --model-dir "${model_dir}" \
        -v
done
