#!/usr/bin/env bash
#SBATCH --job-name=cherimoya_test_install
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:A40|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5|GPU_SKU:L40S
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=akundaje,owners
#SBATCH --output=logs/cherimoya_test_install_%j.out
#SBATCH --error=logs/cherimoya_test_install_%j.err
#
# Smoke-test the native (non-Apptainer) Cherimoya environment on Sherlock:
# every package imports, CUDA works, and a tiny Cherimoya model runs a real
# forward pass. Run setup_env.sh first.
#
# Usage:
#   sbatch src/cherimoya/sherlock_native/test_install.sh

set -euo pipefail

VENV_DIR="${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"

ml load math
ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314
source "${VENV_DIR}/bin/activate"

nvidia-smi -L

python3 - <<'EOF'
import torch
import triton
import bpnetlite
import tangermeme
import h5py
import pandas
import yaml
from cherimoya import Cherimoya

print(f"torch {torch.__version__} (cuda {torch.version.cuda}), triton {triton.__version__}")
assert torch.cuda.is_available(), "CUDA not available"

model = Cherimoya(
    n_filters=8, n_layers=2, signal_groups=[2], n_control_tracks=0,
    trimming=10, verbose=False, compile=False,
).cuda()

x = torch.zeros(2, 4, 500).cuda()
y_profile, y_counts = model(x)
assert y_profile.shape == (2, 2, 480), y_profile.shape
assert y_counts.shape == (2, 1), y_counts.shape
print("OK: forward pass output shapes", tuple(y_profile.shape), tuple(y_counts.shape))
EOF

echo "Cherimoya native install check passed."
