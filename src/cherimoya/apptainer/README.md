# Building the Cherimoya Apptainer Image

This is a Sherlock HPC workaround. Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. The Apptainer image sidesteps that environment issue for the Cherimoya launch scripts.

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
