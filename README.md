# PRO-cap Atlas

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/collections/adamyhe/procap-atlas)
[![Open locus viewer in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kundajelab/procap-atlas/blob/main/notebooks/procap_atlas_bpnet_locus_viewer.ipynb)

Preprocessing, modeling, and atlas-level analysis for ENCODE PRO-cap data.
PRO-cap (Precision Run-On sequencing with cap selection) profiles
transcription start sites and promoter-proximal initiation.

This repository contains the data preparation pipeline, trained-model workflows,
benchmarking scripts, attribution and motif analysis utilities, and UCSC track
hub generation code used for the atlas.

## Status

- **BPNet** is the primary deployed model family. See
  [`src/bpnet/`](src/bpnet/README.md).
- **Cherimoya** models are still in development. See
  [`src/cherimoya/`](src/cherimoya/README.md).
- **MetaFormer / PromoterAI** models are still in development. See
  [`src/metaformer/`](src/metaformer/README.md).
- **ProCapNet** stub directory for some early benchmarks I did. See
  [`src/procapnet/`](src/procapnet/README.md).

## Dependencies

Python dependencies are managed by `uv`. Treat the root `pyproject.toml` and
`uv.lock` as the source of truth for Python packages:

```bash
uv sync --group dev
uv run pytest
```

`pybigtools` is pinned to 0.2.5 for Sherlock compatibility. Torch is not a base
dependency — it lives in two mutually exclusive extras, because Sherlock's
pip/uv can only resolve wheels up to `torch==2.6.0`, while Cherimoya needs its
real `torch>=2.9.0`/`triton>=3.5.1`:

```bash
uv sync --extra sherlock # BPNet / preprocessing (torch==2.6.0)
uv sync --extra cherimoya # native Cherimoya, non-Sherlock Linux hardware only
uv sync --extra hf # huggingface upload
```

**`uv run` flags (`--extra`, `--frozen`, `--project`) must come *before* the
command being run, not after** — `uv run --extra sherlock python script.py`,
not `uv run python script.py --extra sherlock`. Put after the script, `--extra
sherlock` is just forwarded as a literal argument to the script instead of
being read by `uv run`, so no extra actually gets selected.

Use `environment.yml` only to create an optional conda tools environment with
`uv` plus non-Python command-line tools from conda/bioconda:

```bash
mamba env create -f environment.yml
mamba activate procap-atlas
uv sync --group dev
```

Do not add Python package requirements to `environment.yml` or restore a
conda-plus-pip dependency workflow. New Python dependencies belong in
`pyproject.toml` and must be reflected in `uv.lock`.

Non-Apptainer cluster launchers activate `${PROCAP_ATLAS_ENV:-procap-atlas}` by
default to expose `uv` and command-line tools, then run repo Python entrypoints
with `uv run --extra sherlock --frozen`. Set `PROCAP_ATLAS_ENV` before
submission if your environment has a different name:

```bash
export PROCAP_ATLAS_ENV=my-env-name
```

Cherimoya can be difficult to install on certain HPC environments with very old
compilers or custom setups (e.g. Stanford's Sherlock). To get around this, I
build and run the Cherimoya scripts through an Apptainer image on Sherlock (see
[`src/cherimoya/apptainer/`](src/cherimoya/apptainer/README.md)); on other
Linux hardware, use `uv sync --extra cherimoya` directly instead.

`MotifCompendium` and `personal_bpnet` are separate optional/local research
environments, run with plain `python` inside their own conda environment, not
through `uv run`. They are not part of the default `uv` project.
`hubCheck` is also treated as an external UCSC binary because it is not
available from the configured conda channels.

## Pipeline Sketch

Run commands from the repository root. Detailed commands, options, outputs, and
cluster caveats live in the linked workflow READMEs.

```text
download -> preprocess -> train/evaluate models -> attributions/motifs -> tracks/hub
```

Primary workflow docs:

- [`src/download/`](src/download/README.md): reference genome, ENCODE BigWigs,
  peak BED files, and GENCODE annotations.
- [`src/preprocess/`](src/preprocess/README.md): experiment config, merged
  BigWigs, processed peaks, filtered peaks, union peaks, GC negatives, and read
  counts.
- [`src/bpnet/`](src/bpnet/README.md): BPNet training, benchmarking, predicted
  tracks, attributions, MoDISco, motif clustering, and upload.
- [`src/cherimoya/`](src/cherimoya/README.md): Cherimoya training,
  benchmarking, Apptainer notes, and architecture sweeps.
- [`src/hub/`](src/hub/README.md): UCSC track hub generation, bigBed
  conversion, Hugging Face track hosting, and validation.
- [`src/analysis/`](src/analysis/README.md): atlas-level count correlations and
  model warning flags.
- [`src/metaplot/`](src/metaplot/README.md): TSS-centered PRO-cap metaplots and
  heatmaps.
- [`src/metaformer/`](src/metaformer/README.md): PromoterAI / MetaFormer helper
  scripts and cluster templates.
- [`src/procapnet/`](src/procapnet/README.md): legacy ProCapNet benchmarking.

## Repository Layout

```text
data_manifests/   Curated ENCODE URL manifests and metadata
configs/          Generated experiment config, fold splits, read counts, flags
hf/               Hugging Face markdowns and BPNet model metadata
src/download/     Download scripts
src/preprocess/   Processing pipeline for model inputs
src/bpnet/        Primary BPNet model workflow
src/cherimoya/    Cherimoya model workflow
src/hub/          UCSC track hub and hosted track utilities
src/analysis/     Atlas-level analyses and QC flags
src/metaplot/     TSS-centered signal plots
src/metaformer/   Experimental PromoterAI / MetaFormer helpers
src/procapnet/    Legacy ProCapNet benchmark helper
tests/            Unit tests
data/             Gitignored downloaded and processed data
models/           Gitignored trained model artifacts
```

After download and preprocessing, `data/` contains the hg38 reference, raw
ENCODE inputs, processed strand BigWigs, processed peaks, filtered peaks, union
peaks, and GC-matched negatives. See
[`src/download/`](src/download/README.md) and
[`src/preprocess/`](src/preprocess/README.md) for exact paths.

## Public Resources

- Hugging Face collection:
  [`adamyhe/procap-atlas`](https://huggingface.co/collections/adamyhe/procap-atlas)
  - models
  - processed training data
  - prediction and model attribution tracks
  - TF-MoDISco motif calls
- [UCSC track hub](https://genome.ucsc.edu/cgi-bin/hgHubConnect):
  `https://huggingface.co/datasets/adamyhe/procap-atlas-tracks/resolve/main/ucsc/hub.txt`
