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
  - `fit/fit_bpnet.py` — Consolidated BPNet training script with configurable background sampling. Accepts repeatable `--background NAME:RATIO` arguments (names: `ccre`, `gc`) where RATIO is negatives-per-positive contributed by that source. Sources are pooled proportionally. Default: `gc:1/7` (no cCREs; `negative_ratio=1/7`, giving 1/8 of each batch as negatives). Output directory defaults to `models/bpnet/{experiment}`; when `--background` is explicitly specified the suffix encodes the config (e.g. `{experiment}_gc0.1`).
  - `fit/launch.py` — Submits SLURM jobs to train BPNet models, one per (experiment, fold) pair. Skips experiments with fewer than `--min-reads` total reads (default: 10M) and folds where the model file already exists. Extra arguments forwarded to `fit_bpnet.py` via `--fit-args`. Logs to `logs/bpnet/`.
  - `benchmark/benchmark_bpnet.py` — Evaluates a trained BPNet across all folds on held-out test chromosomes. Defaults to `models/bpnet/{experiment}`; use `--model-dir` to override. Reports per-fold and genome-wide profile Pearson, profile JSD, log-counts Pearson, and counts Spearman. Always writes results to `performance_metrics/bpnet/{model_dir_name}.json`.
  - `attribute/attribute_bpnet.py` — Computes DeepLIFT/SHAP attributions across all folds for a trained BPNet, averaging fold attributions genome-wide. Defaults to `models/bpnet/{experiment}`; use `--model-dir` to override. `--head` selects profile or count head. Does not require signal BigWigs (peak loci only). Output: `attributions/bpnet/{model_dir_name}_{head}.npz`.
  - `attribute/save_ohe.py` — Extracts and saves one-hot-encoded sequences for all peaks of a given experiment to `attributions/bpnet/{experiment}_ohe.npz`. Run separately from attributions.
  - `attribute/run_ohe.py` — Runs `save_ohe.py` for all experiments asynchronously. Skips experiments where the OHE file already exists; `-j/--jobs` controls concurrency (default: 4); `--min-reads` and `--dry-run` supported.
  - `attribute/launch.py` — Submits SLURM jobs for attributions, one per (experiment, head) pair. Skips experiments with incomplete training (any fold model missing) or existing output; `--min-reads` filter defaults to 0 (disabled). `--head` is repeatable (default: `profile`). Logs to `logs/bpnet_attr/`.
  - `modisco/launch.py` — Submits SLURM jobs for tf-modisco, one per (experiment, head) pair. Skips experiments where the attribution or OHE npz is missing or modisco output already exists. Runs `modisco motifs` then `modisco report` in a single job. Output: `modisco/bpnet/{exp_id}_{head}.modisco.h5` and `modisco/bpnet/{exp_id}_{head}.modisco/`. Logs to `logs/bpnet_modisco/`.
- **`src/cherimoya/`** — Cherimoya deep learning model training and evaluation (similar API to BPNet):
  - `fit/fit_cherimoya.py` — **Production** Cherimoya training script. Uses `CheriBlock` with a Triton fused dilated conv+norm kernel. Muon + AdamW dual-optimizer setup with warmup + cosine decay schedules. Accepts the same `--background NAME:RATIO` pattern as BPNet; `--muon-lr` and `--adam-lr` set the respective learning rates. Output directory defaults to `models/cherimoya/{experiment}`; non-default backgrounds append a suffix. Logs to `logs/cherimoya_fit/`.
  - `fit/cherimoya2.py` — Pure PyTorch `CheriBlock2` implementation (replaces the fused Triton kernel with `Conv1d` + `LayerNorm`). **For testing/development only** — not intended for production training runs.
  - `fit/fit_cherimoya2.py` — Training script for the pure PyTorch `CheriBlock2` model. Same interface as `fit_cherimoya.py`. **For testing/development only.**
  - `fit/data_loader.py` — `PeakGenerator` and `PeakNegativeSampler` for strand-specific PRO-cap data loading during Cherimoya training.
  - `fit/launch.py` — Submits SLURM jobs to train Cherimoya models, one per (experiment, fold) pair. Skips experiments with fewer than `--min-reads` total reads (default: 0) and folds where the model file already exists. Extra arguments forwarded via `--fit-args`. Logs to `logs/cherimoya_fit/`.
  - `benchmark/benchmark_cherimoya.py` — Evaluates a trained Cherimoya across all folds on held-out test chromosomes. Defaults to `models/cherimoya/{experiment}`; use `--model-dir` to override. Reports per-fold and genome-wide profile Pearson, profile JSD, log-counts Pearson, and counts Spearman. Always writes results to `performance_metrics/cherimoya/{model_dir_name}.json`. `--save-output` writes scaled predictions to `predictions/cherimoya/{model_dir_name}.npz`.
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
python src/bpnet/fit/launch.py                     # submit all experiments x 7 folds via SLURM
python src/bpnet/fit/launch.py --dry-run           # preview sbatch scripts without submitting
```

### Cherimoya Training

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0                   # train Cherimoya, default backgrounds
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1  # GC negatives only
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --muon-lr 0.005 --adam-lr 0.002
python src/cherimoya/fit/launch.py                                                    # submit all experiments x 7 folds via SLURM
python src/cherimoya/fit/launch.py --dry-run                                          # preview sbatch scripts without submitting
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM                 # benchmark across all folds
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output   # also save predictions
```

### Attributions

```bash
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM              # profile head (default)
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head count  # count head
python src/bpnet/attribute/save_ohe.py -e ENCSR882DWM                      # save one-hot sequences separately
python src/bpnet/attribute/run_ohe.py                                       # save OHE for all experiments (4 concurrent)
python src/bpnet/attribute/run_ohe.py -j 8 --min-reads 10000000            # 8 concurrent, skip low-coverage
python src/bpnet/attribute/launch.py                                        # submit all via SLURM (profile head)
python src/bpnet/attribute/launch.py --head profile --head count            # submit both heads
python src/bpnet/attribute/launch.py --dry-run                              # preview without submitting
```

### Modisco

```bash
python src/bpnet/modisco/launch.py                                          # submit all experiments, profile head
python src/bpnet/modisco/launch.py --head profile --head count              # both heads
python src/bpnet/modisco/launch.py --dry-run                                # preview without submitting
```

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `bgzip` (htslib) — bgzip compression of processed peak and negative BED files
- `parallel` (GNU parallel) — parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` (UCSC Kent tools) — BigWig merging
- `sbatch` (SLURM, optional) — cluster job submission via `src/bpnet/fit/launch.py` and `src/bpnet/attribute/launch.py`
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
