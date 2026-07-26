#!/usr/bin/env bash
# Bootstrap the native (non-Apptainer) Cherimoya environment on Sherlock,
# using SRCC's own py-pytorch/py-triton modules instead of an NGC container.
#
# Background: nvcr.io/nvidia/pytorch:26.05-py3 (the Apptainer image under
# src/cherimoya/apptainer/) bundles CUDA 13.2, but Sherlock's GPU driver caps
# out at CUDA 12.4 -- a major-version gap that NVIDIA's forward-compatibility
# mechanism can't bridge, so that image cannot run on Sherlock at all
# (confirmed across multiple GPU SKUs). py-pytorch/2.9.1_py314 is compiled
# against CUDA 12.6, a same-major-version gap that the driver *does* bridge;
# confirmed working (torch.zeros(1).cuda()) on both L40S and A30.
#
# This creates a venv on top of that module pair and installs everything else
# Cherimoya needs into it. --system-site-packages lets the venv still see
# Lmod's PYTHONPATH-injected torch/triton (and whatever py-pytorch's own
# Spack dependency tree provides, e.g. numpy) -- PYTHONPATH is honored by
# Python regardless of venv isolation, which only isolates site-packages
# resolution. Run once per user account; re-run to rebuild the venv from
# scratch if it's ever corrupted or the scratch space is purged.
#
# Usage:
#   bash src/cherimoya/sherlock_native/setup_env.sh

set -euo pipefail

VENV_DIR="${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"

ml load math
ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314

mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# pybigtools (a hard tangermeme dependency) has no Python 3.14 wheel on PyPI
# at any released version, and its latest PyPI release (0.3.0) fails to build
# from source under 3.14 because it pins pyo3==0.22, which caps out at Python
# 3.13. Upstream bumped to pyo3 0.28 (which supports 3.14) on its unreleased
# master branch; install that commit first so pip's resolver sees
# tangermeme's `pybigtools>=0.2` already satisfied and leaves it alone below.
#
# CC=gcc: `ml load math` sets CC=mpicc, whose wrapped flags make its
# underlying compiler emit AVX-512 VNNI instructions (e.g. vpdpbusd) when
# building pybigtools' libdeflate-sys dependency; the system assembler on at
# least some Sherlock nodes is too old to recognize them and fails with
# "no such instruction" / "junk at end of line" errors. Plain gcc doesn't
# bake in those flags and builds cleanly.
CC=gcc python3 -m pip install \
    "pybigtools @ git+https://github.com/jackh726/bigtools.git@34e0a82ee9af2f4f6ebd3268ac692f64e839f100#subdirectory=pybigtools"

# numpy is intentionally not installed here: py-pytorch already provides one
# via --system-site-packages, and a second pip-installed copy could shadow
# it and break torch's ABI.
python3 -m pip install \
    scipy pandas h5py tqdm seaborn modisco tangermeme bam2bw joblib pyyaml

# --no-deps: both declare an unconditional macs3 dependency that nothing in
# this repo calls, and would otherwise also try to replace the module-
# provided torch/triton with PyPI builds.
python3 -m pip install --no-deps bpnet-lite cherimoya==0.2.0

echo "Cherimoya native Sherlock environment ready at $VENV_DIR"
echo "Activate it in future sessions with:"
echo "  ml load math"
echo "  ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314"
echo "  source $VENV_DIR/bin/activate"
