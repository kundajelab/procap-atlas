# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Architecture

- **`data_manifests/`** — Curated lists of ENCODE file URLs, experiment metadata (TSV), and an archive blacklist. These drive the download and preprocessing scripts.
- **`src/download/`** — Bash scripts to fetch reference genome, BigWig signal tracks, and bidirectional peak BED files from ENCODE.
- **`src/preprocess/`** — Python preprocessing scripts (require `pyyaml`):
  - `generate_config.py` — Cross-references experiment report TSV with manifest URL files to produce `configs/experiment_config.yaml`. Excludes archived file IDs via `archive_blacklist.txt`. Falls back to divergent peaks when bidirectional peaks are missing (with warnings). Includes processed output paths for each experiment.
  - `merge_bigwigs.py` — Merges replicate BigWig files per experiment (strands kept separate) into `data/processed/bigwigs/`. Single-replicate files are copied. Requires UCSC Kent tools (`bigWigMerge`, `bedGraphToBigWig`).
  - `rename_peaks.py` — Moves and renames peak BED files to `data/processed/peaks/` with experiment ID, biosample, and peak type in the filename.
- **`configs/`** — Generated YAML experiment config (produced by `generate_config.py`).
- **`src/bpnet/`** — BPNet deep learning model (planned).
- **`src/alphagenome/`** — AlphaGenome analysis (planned).
- **`data/`** (gitignored, created by scripts) — Downloaded genome, signal files, and peaks.

## Environment Setup

Requires Python 3.10+. Sample installation with mamba:

```bash
mamba create -n procap-atlas python=3.12 -y
mamba activate procap-atlas
mamba install -c bioconda -c conda-forge samtools ucsc-bigwigmerge ucsc-bedgraphtobigwig -y
pip install -r requirements.txt
```

## Pipeline Commands

Scripts must run in order: download → generate config → merge/rename.

### Data Download

```bash
bash src/download/download_genome.sh      # GRCh38 reference -> data/hg38.fa, .fai, .chrom.sizes
bash src/download/download_bigwigs.sh     # Plus/minus strand BigWigs (8 parallel jobs) -> data/raw/bigwigs/
bash src/download/download_peaks.sh       # Peak BED files (4 parallel jobs) -> data/raw/peaks/
```

### Preprocessing

Preprocessing scripts are typically run from the repo root, but can be run from anywhere:

```bash
python src/preprocess/generate_config.py   # Manifest → configs/experiment_config.yaml
python src/preprocess/merge_bigwigs.py     # Merge replicate BigWigs (-j/--jobs flag, default 4 workers)
python src/preprocess/rename_peaks.py      # Rename and organize peak files
```

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `parallel` (GNU parallel) — parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` (UCSC Kent tools) — BigWig merging
- `pyyaml`, `tqdm` (Python) — YAML handling and progress bars

## Data Layout After Download and Preprocessing

```
data/
├── hg38.fa
├── hg38.fa.fai
├── hg38.chrom.sizes
├── raw/
│   ├── bigwigs/    # strand-specific signal tracks (per replicate)
│   └── peaks/      # bidirectional peak coordinates
└── processed/
    ├── bigwigs/    # merged across replicates: {experiment}_{biosample}_{strand}.bigWig
    └── peaks/      # renamed: {experiment}_{biosample}_{peak_type}.bed.gz
```
