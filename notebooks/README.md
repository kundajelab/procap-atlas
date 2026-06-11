# Notebooks

## BPNet Locus Diagnostics

`procap_atlas_bpnet_locus_viewer.ipynb` generates locus predictions and
diagnostics for DeepLIFT reference-sequence instability. It downloads the
selected experiment's models, observed tracks, metadata, and hg38 reference
genome rather than using precomputed predicted tracks.

The diagnostic figures include joint-profile and strand-specific profile
DeepLIFT attributions. Each strand-specific target uses one softmax over both
strands before summing the weighted logits for the selected strand. The
notebook also plots centered logits and count-scaled profiles for the two
shuffled-reference selections from every seed: the strongest fold-averaged
20 bp activity and the reference closest to that seed's median activity.

The notebook is designed to run on a Sherlock GPU through Open OnDemand
JupyterLab using the repository's `uv` environment.

### Register the uv kernel

Run this once from a Sherlock login shell:

```bash
mamba activate procap-atlas
cd /path/to/procap-atlas

uv sync --group notebook
uv run --group notebook python notebooks/install_uv_kernel.py
```

The notebook group pins `pyzmq==26.4.0`. That release provides a CPython 3.12
manylinux2014 wheel compatible with Sherlock. Newer releases without that wheel
may build against Sherlock's system libraries and produce a kernel extension
that incorrectly requires `libcudart.so.12`.

Confirm that Jupyter can discover the kernel:

```bash
uv run --group notebook jupyter kernelspec list
```

The user kernelspec should be installed at:

```text
~/.local/share/jupyter/kernels/procap-atlas/
```

The installed kernelspec uses a small launcher in the kernelspec directory. It
removes Open OnDemand's injected `PYTHONPATH`, disables user site packages, and
changes to the repository root before starting this checkout's
`.venv/bin/python`. Starting from the repository root makes imports such as
`src.bpnet.attribute.locus_diagnostics` available without adding the checkout
to `PYTHONPATH`. Kernel startup output is written to:

```text
~/.local/state/procap-atlas/kernel.log
```

Reinstall the kernelspec if the repository is moved or its `.venv` is recreated
at a different path.

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

### Experimental low-activity references

`src.bpnet.attribute.locus_diagnostics.low_activity_references` can select a
diverse reference bank from a larger dinucleotide-shuffled candidate pool. It
ranks candidates by their worst percentile across selector folds for predicted
counts, maximum 20 bp count-scaled signal, and profile concentration.

Use held-out selector folds to avoid choosing a model's baseline with the same
model that will be explained:

```python
from src.bpnet.attribute.locus_diagnostics import low_activity_references

result = low_activity_references(
    genomic_input=X,
    selector_models=(
        torch.load(path, map_location="cpu", weights_only=False).eval()
        for fold, path in enumerate(resources["model_paths"])
        if fold != explained_fold
    ),
    n_candidates=10_000,
    n_references=20,
    random_state=REFERENCE_SEEDS[0],
    batch_size=BATCH_SIZE,
    device=DEVICE,
)
references = result["references"]
```

The result also retains all candidate selection scores and fold-level activity
metrics for comparison with the unfiltered reference distribution. This is an
experimental model-aware baseline and should be reported alongside ordinary
random dinucleotide shuffles rather than replacing them silently.

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
  reinstall the kernelspec with `notebooks/install_uv_kernel.py`, restart the
  complete OnDemand session, and select **PRO-cap Atlas (uv)** again.

  `PYTHONPATH` should print as `None`, and the Open OnDemand path should not
  appear in `sys.path`.

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
  `argv` points to
  `~/.local/share/jupyter/kernels/procap-atlas/launch_kernel.sh`.

  Then inspect the startup log:

  ```bash
  tail -n 100 ~/.local/state/procap-atlas/kernel.log
  ```

  Test the isolated environment directly from a terminal on the same compute
  node:

  ```bash
  cd /path/to/procap-atlas
  env -u PYTHONPATH .venv/bin/python -c \
    "import ipykernel, numpy; print(numpy.__version__, numpy.__file__)"
  ```

  If this command fails, the error is in the synced `.venv` rather than the
  Jupyter connection. If it succeeds, restart the complete OnDemand session so
  Jupyter reloads the replaced kernelspec.

- If the startup log reports that `zmq/backend/cython/_zmq` cannot load
  `libcudart.so.12`, replace the locally built `pyzmq` with the pinned wheel:

  ```bash
  cd /path/to/procap-atlas
  uv sync --group notebook --refresh-package pyzmq
  uv run --group notebook python -c \
    "import zmq; print(zmq.__version__, zmq.__file__)"
  uv run --group notebook python notebooks/install_uv_kernel.py
  ```

  Then restart the complete OnDemand JupyterLab session.
- If repository imports fail, make sure the notebook was opened from the
  `procap-atlas` checkout, then reinstall the kernelspec so its launcher records
  the current checkout path:

  ```bash
  uv run --group notebook python notebooks/install_uv_kernel.py
  ```
- If `torch.cuda.is_available()` is false, confirm that the OnDemand job
  requested a GPU and is running on the `gpu` partition.
- If the first run exceeds the requested session time, restart with a longer
  allocation. Completed diagnostics are loaded from the atomic cache instead
  of being recomputed.
