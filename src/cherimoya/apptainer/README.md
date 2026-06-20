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
`https://github.com/jmschrei/cherimoya.git` at tag `v0.1.0`, the latest GitHub
tag available on 2026-06-19.
