# Building the Cherimoya Apptainer Image

**On Sherlock, use [`../sherlock_native/`](../sherlock_native/README.md)
instead.** This image bundles CUDA 13.2, but Sherlock's GPU driver caps out
at CUDA 12.4 — a major-version gap with no compatibility path (see
"Running" below) — so it cannot run on Sherlock at all, confirmed across
multiple GPU SKUs. `sherlock_native/` uses SRCC's own `py-pytorch`/
`py-triton` modules (CUDA 12.6, a same-major-version gap the driver *can*
bridge) instead.

This was originally a Sherlock HPC workaround: Sherlock's compiler and OS are
very old, which makes installing newer packages a major challenge, and the
Apptainer image sidesteps that environment issue for the Cherimoya launch
scripts. It remains useful for other clusters with similarly old compilers
but modern GPU drivers.

Both definitions use `nvcr.io/nvidia/pytorch:26.05-py3`, which includes
NVIDIA's PyTorch 2.12.0a0 build and matching PyTorch ecosystem packages.

Apptainer is not a general requirement for Cherimoya. On another cluster, prefer the native environment setup unless that cluster has similar compatibility constraints.

## Build

```bash
apptainer build cherimoya.sif cherimoya.def
```

`cherimoya.def` installs `cherimoya==0.2.0` from PyPI.

`cherimoya-0.1.0.def` is kept as an archived definition for the previous
`cherimoya==0.1.0` release, in case a model needs to be reproduced with that
version:

```bash
apptainer build cherimoya-0.1.0.sif cherimoya-0.1.0.def
```

The Apptainer definitions install Cherimoya with `--no-deps` after installing
the non-PyTorch dependencies. This preserves the NVIDIA-tested `torch` and
`torchvision` builds already present in `nvcr.io/nvidia/pytorch:26.05-py3`;
letting pip resolve Cherimoya's `torch>=2.9.0` requirement can replace
NVIDIA's `torch 2.12.0a0` build with PyPI Torch and leave companion packages
incompatible.

## Running

Always pass `--writable-tmpfs` alongside `--nv`:

```bash
apptainer exec --nv --writable-tmpfs cherimoya.sif python ...
```

NVIDIA's container entrypoint tries to reset `/usr/local/cuda/compat/lib` at
startup — a forward-compatibility shim that lets an older host GPU driver run
a newer CUDA toolkit than it natively supports. Apptainer's container
filesystem is read-only by default, so that reset fails silently (`rm: cannot
remove '/usr/local/cuda/compat/lib': Read-only file system`), the compat shim
never gets set up, and CUDA init then fails with `RuntimeError: ... Error 803:
system has unsupported display driver / cuda driver combination` on any
Sherlock GPU node whose driver predates this image's CUDA version.
`--writable-tmpfs` gives the container an ephemeral writable overlay so the
entrypoint's setup step can actually complete.

NVIDIA's forward-compatibility mechanism can only bridge so large a gap. If a
node's driver is old enough, CUDA init fails even with `--writable-tmpfs`,
now with a clearer message: `RuntimeError: The NVIDIA driver on your system
is too old (found version NNNNN)`. `nvcr.io/nvidia/pytorch:26.05-py3` bundles
CUDA 13.2.1, which needs a driver newer than any that supports CUDA 12.4 or
older (e.g. `found version 12040` = CUDA 12.4) — there is no PyTorch build
(NGC or PyPI) that has both `torch.optim.Muon` (needs torch>=2.9.0, added
after PyTorch moved to CUDA 13.x) and CUDA 12.4-or-older compatibility, so an
old-driver node genuinely cannot run Cherimoya, regardless of image or wheel
choice.

Sherlock does not expose GPU driver or CUDA version as a queryable SLURM
constraint (only `GPU_BRD`/`GPU_GEN`/`GPU_SKU`/`GPU_MEM`/`GPU_CC`), so there
is no way to target a compatible node ahead of time. This image is currently
confirmed broken on Sherlock (see the banner at the top of this file — use
`sherlock_native/` there instead). `check_gpu.py` submits a short, cheap job
per GPU SKU to test actual compatibility empirically — useful on other
clusters, or to re-check Sherlock if its driver is ever updated:

```bash
python src/cherimoya/apptainer/check_gpu.py --dry-run
python src/cherimoya/apptainer/check_gpu.py
python src/cherimoya/apptainer/check_gpu.py --skus GPU_SKU:L40S GPU_SKU:A100_PCIE
```

Check `logs/cherimoya_check_gpu/*.err` once the jobs finish — a line reading
`OK: moved tensor to cuda:0` means that SKU (or rather, whichever node it
happened to land on) works.

## Troubleshooting

`nvcr.io/nvidia/pytorch:26.05-py3` is a large image (CUDA libraries
included), so the final `mksquashfs` packing step needs meaningful memory and
scratch space:

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
