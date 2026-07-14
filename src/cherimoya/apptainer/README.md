# Building the Cherimoya Apptainer Image

This is a Sherlock HPC workaround. Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. The Apptainer image sidesteps that environment issue for the Cherimoya launch scripts.

Both definitions use `nvcr.io/nvidia/pytorch:26.05-py3`, which includes
NVIDIA's PyTorch 2.12.0a0 build and matching PyTorch ecosystem packages.

Apptainer is not a general requirement for Cherimoya. On another cluster, prefer the native environment setup unless that cluster has similar compatibility constraints.

## Build

```bash
apptainer build cherimoya.sif cherimoya.def
```

`cherimoya.def` installs `cherimoya==0.1.0` from PyPI.

An experimental GitHub-sourced definition is also available:

```bash
apptainer build cherimoya-github.sif cherimoya-github.def
```

`cherimoya-github.def` installs Cherimoya from
`https://github.com/jmschrei/cherimoya.git` at commit
`b7948c1ee6f648b05e50f52c098fcc5e4f0fede9`, where the package version is
`0.1.1`. This is the latest GitHub version available on 2026-06-21; PyPI still
only publishes `0.1.0`.

The Apptainer definitions install Cherimoya with `--no-deps` after installing
the non-PyTorch dependencies. This preserves the NVIDIA-tested `torch` and
`torchvision` builds already present in `nvcr.io/nvidia/pytorch:26.05-py3`;
letting pip resolve Cherimoya's `torch>=2.9.0` requirement can replace
NVIDIA's `torch 2.12.0a0` build with PyPI Torch and leave companion packages
incompatible.
