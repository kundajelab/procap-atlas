# Building the Cherimoya Apptainer Image

**On Sherlock, this image is an untested candidate fix, not a confirmed
path — verify with `check_gpu.py` (see "Running" below) before relying on it
for real jobs.** [`../sherlock_native/`](../sherlock_native/README.md)
remains the *confirmed*-working Sherlock option (SRCC's own `py-pytorch`/
`py-triton` Lmod modules).

`cherimoya.def` used to bootstrap from `nvcr.io/nvidia/pytorch:26.05-py3`
(CUDA 13.2), which is confirmed broken on Sherlock — its GPU driver caps out
at CUDA 12.4, a major-version gap NVIDIA's forward-compatibility shim can't
bridge (see "Historical note" below). It now bootstraps from the official
`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` Docker Hub image instead,
suggested by SRCC support: CUDA 12.6 is the same version Sherlock's
`py-pytorch/2.9.1_py314` module uses and has already confirmed works via the
driver's compatibility mechanism (see
[`../sherlock_native/README.md`](../sherlock_native/README.md)), and torch
2.13.0 comfortably satisfies Cherimoya's `torch>=2.9.0` requirement (needed
for `torch.optim.Muon`). This hasn't yet been build-and-run-tested on real
Sherlock hardware, though, so treat it as promising rather than proven until
`check_gpu.py` (or a real training job) confirms it.

This was originally a Sherlock HPC workaround: Sherlock's compiler and OS are
very old, which makes installing newer packages a major challenge, and the
Apptainer image sidesteps that environment issue for the Cherimoya launch
scripts. It remains useful for other clusters with similarly old compilers.

Apptainer is not a general requirement for Cherimoya. On another cluster,
prefer the native environment setup unless that cluster has similar
compatibility constraints.

## Build

SRCC support notes that building directly on Sherlock is unreliable
(`apptainer build` there is "a bit spotty"); building on a separate machine
that has Apptainer installed and transferring the resulting `.sif` is the
suggested path, though building directly on Sherlock with the workarounds in
"Troubleshooting" below is also an option:

```bash
# On a separate machine with Apptainer installed:
apptainer build cherimoya.sif cherimoya.def
scp cherimoya.sif sherlock:/path/to/destination/

# Or directly on Sherlock (see "Troubleshooting" if this fails):
apptainer build cherimoya.sif cherimoya.def
```

`cherimoya.def` installs `cherimoya==0.2.0` from its `v0.2.0` git tag rather
than the PyPI wheel (equivalent, since cherimoya has no compiled extensions —
just avoids a PyPI round-trip).

`cherimoya-0.1.0.def` is kept as an archived definition for the previous
`cherimoya==0.1.0` release (still on the old `nvcr.io/nvidia/pytorch:26.05-py3`
base image, unchanged), in case a model needs to be reproduced with that
version:

```bash
apptainer build cherimoya-0.1.0.sif cherimoya-0.1.0.def
```

The Apptainer definitions install Cherimoya with `--no-deps` after installing
the non-PyTorch dependencies. This preserves the `torch`/`torchvision` build
already present in the base image; letting pip resolve Cherimoya's
`torch>=2.9.0` requirement can replace it with a different PyPI Torch build
and leave companion packages incompatible.

Unlike the old `nvcr.io/nvidia/pytorch:26.05-py3` image (a conda-managed
Python), `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` is Ubuntu
24.04-based and uses that distro's system Python, which PEP 668 protects
from bare `pip install` (`error: externally-managed-environment`). Every
`pip install` in `cherimoya.def` passes `--break-system-packages` for this
reason — safe here since the image is single-purpose and not a shared
system pip could conflict with apt over.

## Running

```bash
apptainer exec --nv cherimoya.sif python ...
```

The previous `nvcr.io/nvidia/pytorch:26.05-py3` base image needed
`--writable-tmpfs` alongside `--nv` — its NVIDIA-authored container
entrypoint tries to reset `/usr/local/cuda/compat/lib` at startup (a
forward-compatibility shim letting an older host driver run a newer CUDA
toolkit), which silently fails on Apptainer's read-only-by-default
filesystem and leaves CUDA init failing with `RuntimeError: ... Error 803:
system has unsupported display driver / cuda driver combination`.
`pytorch/pytorch`'s official images don't carry that NVIDIA-specific
entrypoint, so this likely doesn't apply here — but if a similar CUDA-init
error shows up, try adding `--writable-tmpfs` back as a first troubleshooting
step.

Sherlock does not expose GPU driver or CUDA version as a queryable SLURM
constraint (only `GPU_BRD`/`GPU_GEN`/`GPU_SKU`/`GPU_MEM`/`GPU_CC`), so there
is no way to target a compatible node ahead of time, and no way to check
compatibility other than actually running something. `check_gpu.py` submits
a short, cheap job per GPU SKU to test actual compatibility empirically —
run this before trusting the image on Sherlock, and on other clusters, or to
re-check Sherlock if its driver is ever updated:

```bash
python src/cherimoya/apptainer/check_gpu.py --dry-run
python src/cherimoya/apptainer/check_gpu.py
python src/cherimoya/apptainer/check_gpu.py --skus GPU_SKU:L40S GPU_SKU:A100_PCIE
```

Check `logs/cherimoya_check_gpu/*.err` once the jobs finish — a line reading
`OK: moved tensor to cuda:0` means that SKU (or rather, whichever node it
happened to land on) works.

## Historical note

The image originally bootstrapped from `nvcr.io/nvidia/pytorch:26.05-py3`,
which bundles CUDA 13.2.1 — a driver-compatibility gap NVIDIA's
forward-compatibility mechanism can't bridge on Sherlock. `found version
12040` (CUDA 12.4, the newest Sherlock driver at the time) is too old a
driver for CUDA 13.2.1, even with `--writable-tmpfs`, giving `RuntimeError:
The NVIDIA driver on your system is too old`. At the time, no PyPI or NGC
PyTorch build combined `torch.optim.Muon` (`torch>=2.9.0`, added after
PyTorch moved its default builds to CUDA 13.x) with CUDA 12.4-or-older
compatibility — PyPI's own `cu124` wheel index tops out at `torch==2.6.0`.
This was confirmed broken across multiple GPU SKUs (`A30`, `L40S`), which is
what motivated [`../sherlock_native/`](../sherlock_native/README.md).
`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` resolves this the same way
`sherlock_native/`'s `py-pytorch/2.9.1_py314` module does: a CUDA 12.6 build
new enough for Muon, old enough for the driver's compatibility mechanism to
bridge.

## Troubleshooting

The base image is large (CUDA libraries included), so the final `mksquashfs`
packing step needs meaningful memory and scratch space when building
directly on Sherlock:

- `FATAL: ... mksquashfs command failed: exit status 139` is a segfault, not a
  build-script error; on a memory-constrained host (e.g. a 16G dev node) it
  usually means `mksquashfs`'s default multi-threaded xz compression ran out
  of RAM. Cap its memory/processor use before building:

  ```bash
  export APPTAINER_MKSQUASHFS_MEM=2G
  export APPTAINER_MKSQUASHFS_PROCS=1
  apptainer build cherimoya.sif cherimoya.def
  ```

- If the build instead runs out of disk space while staging the rootfs
  (default `/tmp`, often small on login nodes), point the temp/cache dirs at
  scratch space. The path **must be absolute** and must already exist —
  a relative path (e.g. `./apptainer_tmp`) fails with
  `FATAL: Unable to create build: failed to find mount point for
  ./apptainer_tmp: no parent mount point found`, since Apptainer's fakeroot
  build needs to resolve which real filesystem mount the tmpdir is on:

  ```bash
  export APPTAINER_TMPDIR="$(pwd)/apptainer_tmp"   # or an absolute scratch path
  export APPTAINER_CACHEDIR="$(pwd)/apptainer_cache"
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  apptainer build cherimoya.sif cherimoya.def
  ```

  The `INFO: User not listed in /etc/subuid, trying root-mapped namespace`
  line that can appear alongside this is a benign fallback notice (no
  subuid/subgid mapping granted on that host), not the cause of a failed
  build.
