# Notebooks

## BPNet Locus Viewer

`procap_atlas_bpnet_locus_viewer.ipynb` is the lightweight frequency-reference
viewer for one locus. It downloads the selected BPNet fold models, observed
plus/minus BigWigs, metadata, and hg38, then:

- generates fold-averaged predicted plus/minus signal locally;
- stacks observed and predicted strand-specific signal in separate panels, with
  the minus strand shown as negative values, so scale differences are easier to
  inspect;
- computes profile-head and count-head DeepLIFT/SHAP logos using the production
  observed-nucleotide-frequency soft reference and places the logos in the same
  summary figure as the tracks. All panels share the same genomic x tick
  positions.

Use this notebook when you want a direct locus visualization without the
shuffled-reference instability diagnostics below. Configure `EXP_ID`, `REGION`,
`REVERSE_COMPLEMENT`, and `TRACK_VALUE_CLIP` in the configuration cell. The
single `REGION` is used for model-input centering, the observed/predicted track
window, and the DeepLIFT logo crop. `TRACK_VALUE_CLIP` controls display-only
symmetric clipping of observed and predicted tracks, for example `200` for
`[-200, 200]` or `None` for the full range.

Open the notebook through the Colab badge in the first cell for the default
workflow. The setup cell detects Colab, clones this repository into
`/content/procap-atlas`, installs the notebook runtime dependencies with `pip`,
and changes the working directory to the checkout. Outside Colab, the setup
cell leaves the current checkout unchanged.

Sherlock/Open OnDemand execution is still supported as an option. Use the
registered `PRO-cap Atlas (uv)` kernel described below and set
`RUN_ONDEMAND_ENV_CHECK = True` in the optional notebook check cell if you want
to verify that Open OnDemand's injected Python paths have been removed.

## BPNet Locus Diagnostics

`procap_atlas_bpnet_locus_diagnostics.py` generates locus predictions and
diagnostics for DeepLIFT reference-sequence instability. It downloads the
selected experiment's models, observed tracks, metadata, and hg38 reference
genome rather than using precomputed predicted tracks. Use this batch script
when you need the full shuffled-reference diagnostic panel set.

The diagnostic figures include joint-profile and strand-specific profile
DeepLIFT attributions. Each strand-specific target uses one softmax over both
strands before summing the weighted logits for the selected strand. The
notebook also plots centered logits and count-scaled profiles for the two
shuffled-reference selections from every seed: the strongest fold-averaged
20 bp activity and the reference closest to that seed's median activity.

The experimental completeness-preserving weighting workflow operates on
`WEIGHTED_REFERENCE_SEED`. It retains per-reference plus/minus DeepLIFT values
and compares uniform weights with two shared weighting schemes:

- profile-only weights penalize high 20 bp probability mass and plus/minus
  profile-mass imbalance;
- profile-plus-count weights additionally penalize high predicted counts.

Metrics are robustly scaled within the selected reference bank, and
`REFERENCE_WEIGHT_TEMPERATURE` controls how strongly contaminated references
are downweighted. Identical weights are applied to both strand targets, so the
combined attribution remains complete relative to one weighted reference
distribution. The notebook plots full-input completeness residuals; logo
cropping is applied only after aggregation.

The logo diagnostics include fold-averaged DeepLIFT logos for every reference
seed, both for the profile/count heads and for the plus/minus profile targets.
Window ISM is also rendered as a sequence logo by assigning each base the mean
score of all overlapping perturbation windows and placing that score on the
observed genomic nucleotide.

Ranked reference-activity curves are drawn separately for every shuffle seed.
A companion prediction figure shows all fold-averaged references for each seed
as faint signed plus/minus tracks with the seed mean overlaid.

The full diagnostics workflow is designed to run on a Sherlock GPU through Open
OnDemand JupyterLab using the repository's `uv` environment.

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
dependencies come from the separately registered `uv` kernelspec. This should
not take 4-8 hours, but requesting extra gives more flexibility.

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

### Run as a batch script

`procap_atlas_bpnet_locus_diagnostics.py` runs the same workflow without
JupyterLab and saves every plot to disk. Run it as a module from the repository
root so the local `src` modules resolve without modifying `PYTHONPATH`:

```bash
uv run --frozen --group notebook python \
  -m notebooks.procap_atlas_bpnet_locus_diagnostics \
  --experiment ENCSR342WAR \
  --point-region chr2:181680717 \
  --view-region chr2:181680467-181681166 \
  --logo-region chr2:181680467-181681167
```

By default, figures are written under
`figures/bpnet/locus_diagnostics/<experiment>/<point>/`. Use `--output-dir`,
`--format`, and `--dpi` to change the output. The script uses the same
`$SCRATCH/procap_atlas_locus_viewer` downloads and keyed diagnostic caches as
the notebook, so a completed or interrupted notebook run can be reused.

For a Sherlock batch job, save and submit a script like:

```bash
#!/bin/bash -l
#SBATCH --job-name=bpnet_locus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -C "GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=akundaje,owners
#SBATCH --time=08:00:00
#SBATCH --output=logs/bpnet_locus_%j.out
#SBATCH --error=logs/bpnet_locus_%j.err

ml openblas/0.3.28
ml xsimd/8.1.0
ml xz/5.8.1
ml hdf5/1.14.4
ml arrow/22.0.0
ml load py-pyarrow/18.1.0_py312
ml lz4/1.8.0
ml biology
ml htslib
ml ucsc-utils

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"
cd /scratch/users/ayhe/procap-atlas
mkdir -p logs
nvidia-smi -L

uv run --frozen --group notebook python \
  -m notebooks.procap_atlas_bpnet_locus_diagnostics \
  --experiment ENCSR342WAR \
  --point-region chr2:181680717 \
  --view-region chr2:181680467-181681166 \
  --logo-region chr2:181680467-181681167
```

The major runtime controls are `--folds`, `--batch-size`,
`--reference-seeds`, `--references`, `--weighted-reference-seed`, and the
window-ISM options shown by `--help`. Use `--force` to ignore an existing
diagnostic cache.

### Experimental low-activity references

`notebooks.select_reference_pool` implements a two-stage reference pool for one
locus:

1. generate many exact shuffled references for each requested seed
2. score every candidate with every BPNet fold and select the lowest-activity
   references per seed

The selection score penalizes high predicted counts, sharp 5 bp / 20 bp
count-scaled profile peaks, high profile concentration, strong strand
imbalance, and low profile entropy. It does not inspect the genomic-input
attribution, so real strand-specific motifs are not directly downweighted.
The low-activity reference workflow uses `src.bpnet.attribute.deeplift`, which
preserves one-hot validation for the genomic input but allows tensor references
to be soft PFMs.

```bash
uv run --frozen --group notebook python \
  -m notebooks.select_reference_pool \
  --experiment ENCSR342WAR \
  --point-region chr2:181680717 \
  --logo-region chr2:181680467-181681167 \
  --candidate-mode dinucleotide \
  --candidate-seeds 0,1,2,3,4,6,7,42,47,100 \
  --candidates-per-seed 500 \
  --selected-per-seed 20 \
  --batch-size 64 \
  --device cuda \
  --plot-format pdf
```

On Sherlock, pass local paths to avoid downloads when they already exist:

```bash
uv run --frozen --group notebook python \
  -m notebooks.select_reference_pool \
  --experiment ENCSR342WAR \
  --point-region chr2:181680717 \
  --model-dir models/bpnet/ENCSR342WAR \
  --fasta /path/to/hg38.fa \
  --candidate-mode mononucleotide \
  --candidates-per-seed 500 \
  --selected-per-seed 20 \
  --device cuda
```

Outputs are written by default to
`plots/bpnet/reference_pool/<experiment>/<point>/<candidate-mode>/`:

- `selected_references.npz`: selected one-hot references and candidate indices.
- `selected_reference_metrics.tsv`: selected candidate activity summaries.
- `candidate_mean_metrics.tsv`: fold-averaged metrics for all candidates.
- `candidate_fold_metrics.tsv`: per-fold candidate metrics for auditing.
- `selection_summary.json`: arguments, paths, counts, and timing breakdown.
- `selected_deeplift_attributions.npz`: profile/count DeepLIFT arrays for the
  selected references and for a single observed-frequency soft reference.
- `metric_distributions.<format>`: all candidates versus selected references.
- `ranked_reference_metrics.<format>`: per-seed metric ranks with selected
  candidates marked.
- `selected_profile_deeplift_logo.<format>` and
  `selected_count_deeplift_logo.<format>`: aggregate selected-reference
  DeepLIFT logos.
- `selected_seed_deeplift_logos.<format>`: seed-specific selected-reference
  DeepLIFT logos for both heads.
- `frequency_reference_profile_deeplift_logo.<format>` and
  `frequency_reference_count_deeplift_logo.<format>`: DeepLIFT logos and summed
  attribution tracks using one soft reference whose base probabilities are the
  observed input-wide nucleotide frequencies repeated at every position.
- `timing_summary.<format>`: runtime breakdown, including fold scoring.

Use `selection_summary.json` to compare overhead against the ordinary fixed
reference-bank diagnostics. Use `--plot-format png|pdf|svg`, `--logo-window`,
`--logo-region`, `--no-deeplift`, or `--no-plots` to control plotting. This is
an experimental model-aware baseline and should be reported alongside ordinary
random shuffles rather than replacing them silently. The default mode is
dinucleotide shuffling; use `--candidate-mode mononucleotide` to preserve only
input-wide A/C/G/T counts.

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
