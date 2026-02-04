# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Architecture

- **`data_manifests/`** — Curated lists of ENCODE file URLs, experiment metadata (TSV), and an archive blacklist. These drive the download and preprocessing scripts.
- **`src/download/`** — Bash scripts to fetch reference genome, BigWig signal tracks, and bidirectional peak BED files from ENCODE.
- **`src/preprocess/`** — Python preprocessing scripts (require `pyyaml`):
  - `generate_config.py` — Cross-references experiment report TSV with manifest URL files to produce `configs/experiment_config.yaml`. Excludes archived file IDs via `archive_blacklist.txt`. Falls back to divergent peaks when bidirectional peaks are missing (with warnings).
  - `merge_bigwigs.py` — Merges replicate BigWig files per experiment (strands kept separate) into `data/processed/bigwigs/`. Single-replicate files are moved/renamed. Requires UCSC Kent tools (`bigWigMerge`, `bedGraphToBigWig`).
  - `rename_peaks.py` — Moves and renames peak BED files to `data/processed/peaks/` with experiment ID, biosample, and peak type in the filename.
- **`configs/`** — Generated YAML experiment config (produced by `generate_config.py`).
- **`src/bpnet/`** — BPNet deep learning model (planned).
- **`src/alphagenome/`** — AlphaGenome analysis (planned).
- **`data/`** (gitignored, created by scripts) — Downloaded genome, signal files, and peaks.

## Data Download Commands

All download scripts use relative paths and must be run from `src/download/`:

```bash
cd src/download

# Download and index GRCh38 reference genome -> data/hg38.fa, hg38.fa.fai, hg38.chrom.sizes
bash download_genome.sh

# Download plus/minus strand BigWig files (4 parallel jobs) -> data/raw/bigwigs/
bash download_bigwigs.sh

# Download bidirectional peak BED files (4 parallel jobs) -> data/raw/peaks/
bash download_peaks.sh
```

## Preprocessing Commands

All preprocessing scripts are run from the repo root and require `pyyaml` (`pip install pyyaml`):

```bash
# Generate experiment config YAML from data manifests
python src/preprocess/generate_config.py

# Merge replicate BigWig files per experiment (keeps strands separate)
python src/preprocess/merge_bigwigs.py

# Move and rename peak BED files with experiment/biosample names
python src/preprocess/rename_peaks.py
```

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `parallel` (GNU parallel) — parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` (UCSC Kent tools) — BigWig merging
- `pyyaml` (Python) — YAML config generation and reading

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
