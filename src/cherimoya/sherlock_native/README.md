# Native Cherimoya Environment on Sherlock

Runs Cherimoya directly on Sherlock using SRCC's own `py-pytorch`/`py-triton`
Lmod modules instead of the Apptainer image under
[`../apptainer/`](../apptainer/README.md).

## Why this exists

The Apptainer image (`nvcr.io/nvidia/pytorch:26.05-py3`) bundles CUDA 13.2,
but Sherlock's GPU driver caps out at CUDA 12.4 across the SKUs tested
(`A30`, `L40S`) — a major-version gap that NVIDIA's forward-compatibility
shim cannot bridge (see [`../apptainer/README.md`](../apptainer/README.md)'s
"Running" section for the full diagnosis). No PyTorch build, from NGC or
PyPI, has both `torch.optim.Muon` (needs `torch>=2.9.0`) and CUDA
12.4-or-older compatibility — PyPI's own `cu124` wheel index tops out at
`torch==2.6.0`.

Sherlock's `py-pytorch/2.9.1_py314` module sidesteps this: it's compiled
against CUDA 12.6, a same-*major*-version gap the driver's compatibility
mechanism does bridge. Confirmed working (`torch.zeros(1).cuda()` succeeds)
on both `A30` and `L40S`. `py-triton/3.5.1_py314` is a matching module built
for the same Python 3.14, satisfying Cherimoya's `triton>=3.5.1` exactly.

This makes the Apptainer image effectively non-functional on Sherlock as of
this writing; this native path is the one to use there. Apptainer remains
useful for other clusters with old compilers but modern GPU drivers.

## Setup

Run once per user account:

```bash
bash src/cherimoya/sherlock_native/setup_env.sh
```

This loads the module pair, creates a venv at
`/scratch/users/$USER/venvs/cherimoya-sherlock` (override with
`CHERIMOYA_VENV_DIR`) with `--system-site-packages` so it still sees the
Lmod-injected torch/triton, and `pip install`s everything else Cherimoya
needs into it (all pure-Python packages except `bpnet-lite`/`cherimoya`,
which are `--no-deps` installed to avoid pulling in an unwanted `macs3` or
replacing the module-provided torch/triton). Re-run it to rebuild the venv if
it's ever corrupted or the scratch space is purged.

`pybigtools` (a hard dependency of `tangermeme`, which `fit_cherimoya.py`
imports) has no Python 3.14 wheel at any released version on PyPI, and its
latest release (0.3.0) fails to build from source under 3.14 because it pins
`pyo3==0.22`, which only supports up to Python 3.13. Upstream bumped to
`pyo3` 0.28 (which does support 3.14) on its `master` branch, unreleased to
PyPI as of this writing. `setup_env.sh` installs that exact commit from git
before anything else, so pip's resolver sees `tangermeme`'s `pybigtools>=0.2`
requirement already satisfied and doesn't try to build the PyPI release
itself. If pybigtools ships a PyPI release built against a
3.14-compatible pyo3, switch back to a normal pinned PyPI install.

Building pybigtools also needs
`CFLAGS=-DLIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_{AVX512VNNI,VPCLMULQDQ,AVX_VNNI}`
(already set in `setup_env.sh`): its bundled `libdeflate` detects the
*compiler's* support for these instruction sets and compiles them
unconditionally, but doesn't check whether the paired assembler (`binutils`)
can actually encode them, unlike libdeflate's own CMake build, which probes
this and disables affected codepaths automatically. Whatever `gcc` these Lmod
modules put on `PATH` is new enough to target AVX-512 VNNI, but the paired
assembler here isn't, so this fails with `no such instruction: vpdpbusd`
without the flags.

`pillow<12.3.0` and `leidenalg<0.11` are also pinned. Both dropped their
`manylinux2014`/`manylinux_2_17` (glibc 2.17) wheels in newer releases,
leaving only glibc 2.27+-tagged wheels that Sherlock's older glibc can't use;
pip then falls back to a source build that fails on missing system `jpeg`/
`igraph` headers. The pinned older releases still ship a glibc-2.17 wheel and
support Python 3.14 (`pillow` via a real `cp314` wheel; `leidenalg` via a
`cp38-abi3` wheel, forward-compatible through the stable ABI). The `pillow`
pin matches the one already used for Sherlock in the root `pyproject.toml`.

## Verify

```bash
sbatch src/cherimoya/sherlock_native/test_install.sh
```

Submits a short (~minutes), cheap job that loads the modules, activates the
venv, imports every package `fit_cherimoya.py` needs, and runs a tiny
Cherimoya model through a real forward pass on the GPU. Check
`logs/cherimoya_test_install_*.out` — `Cherimoya native install check passed`
at the end means the whole environment works end to end. Run this after
`setup_env.sh` and before committing to a full training job.

## Running

Load the modules and activate the venv in any job or interactive session
before running Cherimoya scripts:

```bash
ml load math
ml load py-pytorch/2.9.1_py314 py-triton/3.5.1_py314
source /scratch/users/$USER/venvs/cherimoya-sherlock/bin/activate
python3 src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0
```

`src/cherimoya/fit/launch.py` and the other SLURM launchers under
`src/cherimoya/` generate sbatch scripts that do exactly this.

## Caveats

- The venv's ability to see the Lmod-injected torch/triton via
  `--system-site-packages` + `PYTHONPATH` has not been verified on Sherlock
  directly yet — `test_install.sh` above is exactly that check. If it fails
  to see the module-provided versions, drop `--system-site-packages` from
  `setup_env.sh` and fall back to `pip install` (or `pip install --user`)
  instead.
- Only `A30` and `L40S` have been confirmed working. Other SKUs
  (`A100_PCIE`, `A100_SXM4`, `A40`, `H100_SXM5`, `H200_SXM5`) have not been
  tested against this module pair; if Sherlock's driver is centrally managed
  (uniform across the fleet, as is typical), they should behave the same,
  but this hasn't been verified directly.
- This module pair targets Python 3.14, which is newer than the root `uv`
  project's `requires-python` (`>=3.12,<3.13`). This environment is
  intentionally separate from the root project's `uv`-managed venv; it does
  not go through `uv run` at all.
