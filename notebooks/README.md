# Notebooks

## BPNet Locus Diagnostics

`procap_atlas_bpnet_locus_viewer.ipynb` generates locus predictions and
diagnostics for DeepLIFT reference-sequence instability. It downloads the
selected experiment's models, observed tracks, metadata, and hg38 reference
genome rather than using precomputed predicted tracks.

The notebook is designed to run on a Sherlock GPU through Open OnDemand
JupyterLab using the repository's `uv` environment.

### Register the uv kernel

Run this once from a Sherlock login shell:

```bash
mamba activate procap-atlas
cd /path/to/procap-atlas

uv sync --group notebook
uv run --group notebook python -m ipykernel install --user \
  --name procap-atlas \
  --display-name "PRO-cap Atlas (uv)" \
  --env PYTHONNOUSERSITE 1 \
  --env PYTHONPATH ""
```

Confirm that Jupyter can discover the kernel:

```bash
uv run --group notebook jupyter kernelspec list
```

The user kernelspec should be installed at:

```text
~/.local/share/jupyter/kernels/procap-atlas/
```

The kernelspec points directly to this checkout's `.venv/bin/python`. Reinstall
the kernelspec if the repository is moved or its `.venv` is recreated at a
different path. `PYTHONNOUSERSITE=1` and an empty `PYTHONPATH` prevent the
OnDemand Jupyter server environment from injecting user or system packages into
the project kernel.

### Start the OnDemand session

In Sherlock Open OnDemand, choose **Interactive Apps -> JupyterLab**. Recommended
starting resources for the full seven-fold diagnostic workflow are:

| Setting | Recommended value |
| --- | --- |
| Partition | `gpu` |
| Nodes | `1` |
| CPUs | `4` |
| GPUs | `1` |
| Memory | `32-64 GB` |
| Runtime | `4-8 hours` |
| Working directory, if available | `/path/to/procap-atlas` |

Use the standard Sherlock Jupyter/Python environment in the OnDemand form. The
form environment starts the JupyterLab server; the notebook's Python
dependencies come from the separately registered `uv` kernelspec.

### Select and verify the kernel

Open:

```text
/path/to/procap-atlas/notebooks/procap_atlas_bpnet_locus_viewer.ipynb
```

Select **Kernel -> Change Kernel -> PRO-cap Atlas (uv)**.

Verify the interpreter and GPU from a notebook cell:

```python
import sys
import torch

print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")
```

The interpreter should be:

```text
/path/to/procap-atlas/.venv/bin/python
```

The notebook stores downloads and diagnostic caches under
`$SCRATCH/procap_atlas_locus_viewer` when `$SCRATCH` is available.

### Troubleshooting

- If importing NumPy fails with `libopenblas.so.0: cannot open shared object
  file`, first check which NumPy the kernel is finding without importing it:

  ```python
  import importlib.util
  import os
  import sys

  spec = importlib.util.find_spec("numpy")
  print(sys.executable)
  print(spec.origin)
  print(os.environ.get("PYTHONPATH"))
  print(*sys.path, sep="\n")
  ```

  `spec.origin` must be inside
  `/path/to/procap-atlas/.venv/lib/python3.12/site-packages/`. If it is not,
  reinstall the hardened kernelspec above and restart the complete OnDemand
  JupyterLab session.

  If NumPy is coming from the project `.venv`, update the checkout and resync:

  ```bash
  mamba activate procap-atlas
  cd /path/to/procap-atlas
  uv sync --group notebook
  uv run --group notebook python -c \
    "import numpy; print(numpy.__version__, numpy.__file__)"
  ```

  Restart the kernel after resyncing.
- If the kernel is not listed, stop and restart the OnDemand JupyterLab session
  after registering it.
- If the kernel fails immediately, inspect
  `~/.local/share/jupyter/kernels/procap-atlas/kernel.json` and confirm its
  Python path still exists.
- If repository imports fail, make sure the notebook was opened from the
  `procap-atlas` checkout and the OnDemand working directory is the repository
  root.
- If `torch.cuda.is_available()` is false, confirm that the OnDemand job
  requested a GPU and is running on the `gpu` partition.
- If the first run exceeds the requested session time, restart with a longer
  allocation. Completed diagnostics are loaded from the atomic cache instead
  of being recomputed.
