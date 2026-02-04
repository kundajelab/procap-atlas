# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Architecture

- **`data_manifests/`** — Curated lists of ENCODE file URLs and experiment metadata (TSV). These drive the download scripts.
- **`src/download/`** — Bash scripts to fetch reference genome, BigWig signal tracks, and bidirectional peak BED files from ENCODE.
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

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `parallel` (GNU parallel) — parallel downloads

## Data Layout After Download

```
data/
├── hg38.fa
├── hg38.fa.fai
├── hg38.chrom.sizes
└── raw/
    ├── bigwigs/    # strand-specific signal tracks
    └── peaks/      # bidirectional peak coordinates
```
