# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Overview

This repository supports preprocessing and deep learning-based analysis of the
ENCODE PRO-cap atlas on GRCh38/hg38. PRO-cap (Precision Run-On sequencing with
cap selection) profiles transcription start sites and promoter-proximal
initiation.

The root [`README.md`](README.md) is the project map. Workflow-specific commands,
defaults, outputs, and caveats belong in the relevant subdirectory README.

## Source-of-Truth Docs

- [`src/download/`](src/download/README.md): raw genome, signal, peak, and
  annotation downloads.
- [`src/preprocess/`](src/preprocess/README.md): experiment config generation,
  merged BigWigs, processed/filtered peaks, union peaks, GC negatives, and read
  counts.
- [`src/bpnet/`](src/bpnet/README.md): primary BPNet training, benchmarking,
  predicted visualization tracks, attributions, MoDISco, motif clustering, and
  Hugging Face upload.
- [`src/cherimoya/`](src/cherimoya/README.md): Cherimoya training,
  benchmarking, Apptainer notes, and architecture sweeps.
- [`src/hub/`](src/hub/README.md): UCSC hub generation, bigBed conversion,
  hosted track upload, and validation.
- [`src/analysis/`](src/analysis/README.md): atlas-level count correlations and
  model warning flags.
- [`src/metaplot/`](src/metaplot/README.md): TSS-centered metaplots and
  heatmaps.
- [`src/metaformer/`](src/metaformer/README.md): experimental PromoterAI /
  MetaFormer helpers.
- [`src/procapnet/`](src/procapnet/README.md): legacy ProCapNet benchmarking.

Keep new operational details in these subdirectory READMEs, not in this file or
the root README.

## Dependencies

Python dependencies are managed by the root `pyproject.toml` and `uv.lock`.
New Python dependencies must be added there, not to `environment.yml`.
On Linux/HPC systems, the default non-Cherimoya environment pins Torch to 2.6.0
for Sherlock compatibility. Do not relax that pin for BPNet/preprocessing unless
Sherlock support is explicitly being dropped.
`pybigtools` is pinned to 0.2.5 because newer releases can require source builds
that fail on Sherlock's older assembler/toolchain.

```bash
uv sync --group dev
uv run pytest
```

`environment.yml` is a conda tools environment for `uv` and non-Python
command-line tools:

```bash
mamba env create -f environment.yml
mamba activate procap-atlas
uv sync --group dev
```

Non-Apptainer cluster launchers activate `${PROCAP_ATLAS_ENV:-procap-atlas}` by
default to expose `uv` and command-line tools, then run repo Python entrypoints
with `uv run --frozen`.
Cherimoya should be treated as a separate Apptainer/local environment on
Sherlock because it uses `torch.optim.Muon`.
`MotifCompendium` and `personal_bpnet` are separate optional/local research
environments.
`hubCheck` is treated as an external UCSC binary because it is not available
from the configured conda channels.

## Architecture Notes

- `data_manifests/` contains curated ENCODE URL manifests, experiment metadata,
  and archive blacklist inputs.
- `configs/` contains generated experiment config, chromosome fold splits, read
  counts, warning flags, and related workflow tables.
- `data/`, `models/`, `predictions/`, `attributions/`, `performance_metrics/`,
  `figures/`, and `logs/` are workflow output locations and are generally
  gitignored.
- BPNet is the deployed model family. Cherimoya is trained and benchmarked but
  not deployed. MetaFormer/PromoterAI is experimental. ProCapNet is legacy.
- SLURM launchers and Apptainer scripts are operational templates with
  Sherlock-specific defaults; keep those caveats next to the launchers that use
  them.

## Editing Guidance

- Prefer updating the README nearest the workflow when commands, defaults,
  outputs, or caveats change.
- Keep root docs concise and link to subdirectory docs instead of duplicating
  command inventories.
- Do not update `hf/*.md` model or dataset cards unless the requested change
  explicitly affects public-facing Hugging Face documentation.
- Preserve generated config/data files unless the task explicitly asks to
  regenerate them.
- When changing scripts, keep behavior documented in the matching subdirectory
  README and run the narrowest useful syntax/tests.
