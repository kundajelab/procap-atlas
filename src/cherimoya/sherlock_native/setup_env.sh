#!/usr/bin/env bash
# Bootstrap the native (non-Apptainer) Cherimoya environment on Sherlock,
# using SRCC's own py-pytorch/py-triton modules instead of an NGC container.
#
# Background: the Apptainer image under src/cherimoya/apptainer/ originally
# bootstrapped from nvcr.io/nvidia/pytorch:26.05-py3, which bundles CUDA
# 13.2, but Sherlock's GPU driver caps out at CUDA 12.4 -- a major-version
# gap that NVIDIA's forward-compatibility mechanism can't bridge, so that
# image couldn't run on Sherlock at all (confirmed across multiple GPU
# SKUs). py-pytorch/2.9.1_py314 is compiled against CUDA 12.6, a
# same-major-version gap that the driver *does* bridge; confirmed working
# (torch.zeros(1).cuda()) on both L40S and A30. The Apptainer image has
# since switched to a CUDA 12.6 base image and is also confirmed working
# (see src/cherimoya/apptainer/README.md) -- this native path remains a
# valid fallback (--native on the launchers), not the only working option.
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
# pybigtools bundles libdeflate, which auto-detects the *compiler's* support
# for AVX-512 VNNI/VPCLMULQDQ/AVX-VNNI and unconditionally compiles those
# codepaths if the compiler is new enough -- but doesn't check whether the
# paired *assembler* (binutils) can actually encode the resulting
# instructions, unlike libdeflate's own CMake build, which probes this and
# disables affected codepaths automatically (see libdeflate's CMakeLists.txt,
# `check_assembler_support`). The Rust libdeflate-sys crate's build.rs skips
# that probe, so whatever gcc these Lmod modules put on PATH (new enough to
# target AVX-512 VNNI) gets paired with an older assembler here that can't
# encode it, failing with "no such instruction: vpdpbusd" etc. Passing the
# same -DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_* flags CMake would have
# derived disables those codepaths explicitly; cc-rs forwards CFLAGS to the
# compiler invocation.
CFLAGS="-DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX512VNNI -DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_VPCLMULQDQ -DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX_VNNI" \
    python3 -m pip install \
    "pybigtools @ git+https://github.com/jackh726/bigtools.git@34e0a82ee9af2f4f6ebd3268ac692f64e839f100#subdirectory=pybigtools"

# numpy is intentionally not installed here: py-pytorch already provides one
# via --system-site-packages, and a second pip-installed copy could shadow
# it and break torch's ABI.
#
# pillow<12.3.0 (transitive, via seaborn/matplotlib) and leidenalg<0.11
# (transitive, via modisco): both dropped
# their manylinux2014/manylinux_2_17 (glibc 2.17) wheels in newer releases,
# leaving only glibc 2.27+-tagged wheels that Sherlock's older glibc can't
# use, which falls back to a source build missing system jpeg/igraph
# headers. Older releases still ship a glibc-2.17 wheel and support Python
# 3.14 (pillow via a real cp314 wheel; leidenalg via a cp38-abi3 wheel,
# forward-compatible through the stable ABI). Matches the same pillow pin
# already used for Sherlock in the root pyproject.toml.
python3 -m pip install \
    scipy pandas h5py tqdm seaborn modisco tangermeme bam2bw joblib pyyaml \
    "pillow<12.3.0" "leidenalg<0.11"

# --no-deps: both declare an unconditional macs3 dependency that nothing in
# this repo calls, and would otherwise also try to replace the module-
# provided torch/triton with PyPI builds.
#
# cherimoya is pinned to an exact commit rather than PyPI's 0.2.0 release:
# the upstream "v0.2.0" git tag was force-moved to a commit that adds EMA
# weight averaging and switches checkpoint selection to valid_count_corr
# instead of a combined loss, but the PyPI 0.2.0 wheel predates that rewrite
# and still has the old behavior. Pinning the commit directly (matching
# src/cherimoya/apptainer/cherimoya.def) keeps this path on the same
# algorithm instead of silently training under different code.
python3 -m pip install --no-deps bpnet-lite
python3 -m pip install --no-deps "cherimoya @ git+https://github.com/jmschrei/cherimoya.git@8e4283fe56db4a29418c1d8119da3240d7c709ba"

echo "Cherimoya native Sherlock environment ready at $VENV_DIR"
echo "Activate it in future sessions with:"
echo "  ml load math"
echo "  ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314"
echo "  source $VENV_DIR/bin/activate"
