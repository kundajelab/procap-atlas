# Building the Cherimoya Apptainer Image

This is a Sherlock HPC workaround. Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. The Apptainer image sidesteps that environment issue for the Cherimoya launch scripts.

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
  scratch space:

  ```bash
  export APPTAINER_TMPDIR=/scratch/users/$USER/apptainer_tmp
  export APPTAINER_CACHEDIR=/scratch/users/$USER/apptainer_cache
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  ```
