# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Architecture

- **`data_manifests/`** — Curated lists of ENCODE file URLs, experiment metadata (TSV), and an archive blacklist. These drive the download and preprocessing scripts.
- **`src/download/`** — Bash scripts to fetch reference genome, BigWig signal tracks, and bidirectional peak BED files from ENCODE.
- **`src/preprocess/`** — Python preprocessing scripts (require `pyyaml`):
  - `generate_config.py` — Cross-references experiment report TSV with manifest URL files to produce `configs/experiment_config.yaml`. Excludes archived file IDs via `archive_blacklist.txt`. Falls back to divergent peaks when bidirectional peaks are missing (with warnings). Also collects unidirectional peaks as a separate field per experiment. Includes processed output paths for each experiment.
  - `merge_bigwigs.py` — Merges replicate BigWig files per experiment (strands kept separate) into `data/processed/bigwigs/`. Single-replicate files are copied. Requires UCSC Kent tools (`bigWigMerge`, `bedGraphToBigWig`).
  - `process_peaks.py` — Merges bidirectional and unidirectional peak BED files per experiment using `_merge_uni_bi_peaks.py`, writing combined output to `data/processed/peaks/`. Falls back to copying when only one peak type is available.
  - `_merge_uni_bi_peaks.py` — Reads gzipped bidirectional and unidirectional peak BED files, reformats columns, and returns a sorted merged list. Adapted from ProCapNet.
  - `gc_match.py` — Adapted from tangermeme to extract GC-matched negative regions, extended to support thresholding on multiple bigwig files.
  - `gc_match_run.py` — Runs `gc_match.py` for every experiment in the config, writing bgzip-compressed output to `data/processed/negatives/`. Supports multiprocessing via `-j/--jobs` flag (default 1 worker).
  - `count_reads.py` — Counts total reads in the processed BigWig files for each experiment using `pybigtools`. Sums `value * (end - start)` over all intervals via `records()`; minus-strand values are negated in the BigWig so their total is taken as absolute value. Writes TSV to `configs/n_reads.txt` by default (`--tsv` to override).
- **`configs/`** — Generated YAML experiment config (produced by `generate_config.py`) and chromosome fold splits (`chrom_splits.yaml`, 7 folds for cross-validation).
- **`src/bpnet/`** — BPNet deep learning model training and evaluation:
  - `fit/fit_bpnet.py` — Consolidated BPNet training script with configurable background sampling. Accepts repeatable `--background NAME:RATIO` arguments (names: `ccre`, `gc`) where RATIO is negatives-per-positive contributed by that source. Sources are pooled proportionally. Default: `ccre:1/14 gc:1/14` (total `negative_ratio=1/7`, giving 1/8 of each batch as negatives). Output directory name encodes backgrounds and ratios (e.g. `{experiment}_ccre0.0714_gc0.0714`).
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

Scripts must run in order: download → generate config → merge/rename → gc negatives → model training.

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
python src/preprocess/process_peaks.py     # Merge bidirectional and unidirectional peak files
python src/preprocess/gc_match_run.py      # GC-matched negatives (-j/--jobs flag, default 1 worker)
python src/preprocess/count_reads.py       # Count reads per experiment → configs/n_reads.txt
```

### Model Training

```bash
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0                                          # Train BPNet, default backgrounds (ccre+gc at 1/14 each)
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background gc:0.1               # GC negatives only
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background ccre:0.05 --background gc:0.05  # custom ratios
```

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `bgzip` (htslib) — bgzip compression of processed peak and negative BED files
- `parallel` (GNU parallel) — parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` (UCSC Kent tools) — BigWig merging
- `sbatch` (SLURM, optional) — cluster job submission via `src/bpnet/fit/launch.py`
- Python (see `requirements.txt`): `pyyaml`, `tqdm`, `numpy`, `pandas`, `scipy`, `joblib`, `pyfaidx`, `pyfastx`, `pybigtools`, `torch`, `tangermeme`, `bpnetlite`

## Data Layout After Download and Preprocessing

```
data/
├── hg38.fa
├── hg38.fa.fai
├── hg38.chrom.sizes
├── raw/
│   ├── bigwigs/    # strand-specific signal tracks (per replicate)
│   └── peaks/      # bidirectional and unidirectional peak coordinates
└── processed/
    ├── bigwigs/    # merged across replicates: {experiment}_{biosample}_{strand}.bigWig
    └── peaks/      # merged bidirectional + unidirectional: {experiment}_{biosample}.bed.gz
```
