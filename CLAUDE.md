# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Architecture

- **`data_manifests/`** — Curated lists of ENCODE file URLs, experiment metadata (TSV), and an archive blacklist. These drive the download and preprocessing scripts.
- **`src/download/`** — Bash scripts to fetch the reference genome, BigWig signal tracks, peak BED files from ENCODE, and GENCODE annotations for TSS metaplots.
- **`src/preprocess/`** — Python preprocessing scripts (require `pyyaml`):
  - `generate_config.py` — Cross-references experiment report TSV with manifest URL files to produce `configs/experiment_config.yaml`. Excludes archived file IDs via `archive_blacklist.txt`. Falls back to divergent peaks when bidirectional peaks are missing (with warnings). Also collects unidirectional peaks as a separate field per experiment. Includes processed output paths for each experiment.
  - `merge_bigwigs.py` — Merges replicate BigWig files per experiment (strands kept separate) into `data/processed/bigwigs/`. Single-replicate files are copied. Requires UCSC Kent tools (`bigWigMerge`, `bedGraphToBigWig`).
  - `process_peaks.py` — Merges bidirectional and unidirectional peak BED files per experiment using `_merge_uni_bi_peaks.py`, writing combined output to `data/processed/peaks/`. Falls back to copying when only one peak type is available.
  - `_merge_uni_bi_peaks.py` — Reads gzipped bidirectional and unidirectional peak BED files, reformats columns, and returns a sorted merged list. Adapted from ProCapNet.
  - `_filter_nonACGT_regions.py` — Helper for dropping peak loci whose model input window contains non-ACGT sequence.
  - `filter_peaks_run.py` — Runs the non-ACGT peak filter for every experiment, writing `processed.filtered_peaks`.
  - `_gc_match.py` — Adapted from tangermeme to extract GC-matched negative regions, extended to support thresholding on multiple bigwig files.
  - `gc_match_run.py` — Runs `_gc_match.py` for every experiment in the config, writing bgzip-compressed output to `data/processed/negatives/`. Supports multiprocessing via `-j/--jobs` flag (default 1 worker).
  - `make_union_peaks.py` — Pools processed peak files across experiments, optionally filters by `--min-reads`, applies optional `--slop`, and writes `data/processed/peaks/union_peaks.bed.gz`.
  - `count_reads.py` — Counts total reads in the processed BigWig files for each experiment using `pybigtools`. Sums `value * (end - start)` over all intervals via `records()`; minus-strand values are negated in the BigWig so their total is taken as absolute value. Writes TSV to `configs/n_reads.txt` by default (`--tsv` to override).
- **`configs/`** — Generated YAML experiment config (produced by `generate_config.py`) and chromosome fold splits (`chrom_splits.yaml`, 7 folds for cross-validation).
- **`config.json`** — Root-level Hugging Face model metadata for the BPNet atlas. Keep this at the model repo root when uploading; Hugging Face uses default query files such as `config.json` for model download tracking.
- **`src/bpnet/`** — BPNet deep learning model training and evaluation:
  - `fit/fit_bpnet.py` — Consolidated BPNet training script with configurable background sampling. Accepts repeatable `--background NAME:RATIO` arguments (names: `ccre`, `gc`) where RATIO is negatives-per-positive contributed by that source. Sources are pooled proportionally. Default: `gc:1/7` (no cCREs; `negative_ratio=1/7`, giving 1/8 of each batch as negatives). Output directory defaults to `models/bpnet/{experiment}`; when `--background` is explicitly specified the suffix encodes the config (e.g. `{experiment}_gc0.1`).
  - `fit/launch.py` — Submits SLURM jobs to train BPNet models, one per (experiment, fold) pair. Skips experiments with fewer than `--min-reads` total reads (default: 10M) and folds where the model file already exists. Extra arguments forwarded to `fit_bpnet.py` via `--fit-args`. Logs to `logs/bpnet/`.
  - `benchmark/benchmark_bpnet.py` — Evaluates a trained BPNet across all folds on held-out test chromosomes. Defaults to `models/bpnet/{experiment}`; use `--model-dir` to override. Reports per-fold and genome-wide profile Pearson, profile JSD, log-counts Pearson, and counts Spearman. Always writes results to `performance_metrics/bpnet/{model_dir_name}.json`.
  - `model_upload.py` — Uploads BPNet model artifacts, `configs/`, and root `config.json` to the Hugging Face model repo `adamyhe/procap-atlas`. The root `config.json` enables standard Hugging Face model download tracking for this custom BPNet collection.
  - `attribute/attribute_bpnet.py` — Computes DeepLIFT/SHAP attributions across all folds for a trained BPNet, averaging fold attributions genome-wide over `processed.filtered_peaks`. Defaults to `models/bpnet/{experiment}`; use `--model-dir` to override. `--head` selects profile or count head. Does not require signal BigWigs (peak loci only). Output: `attributions/bpnet/{model_dir_name}_{head}.npz`.
  - `attribute/save_ohe.py` — Extracts and saves one-hot-encoded sequences for filtered peaks of a given experiment to `attributions/bpnet/{experiment}_ohe.npz`. Run separately from attributions.
  - `attribute/run_ohe.py` — Runs `save_ohe.py` for all experiments asynchronously. Skips experiments where the OHE file already exists; `-j/--jobs` controls concurrency (default: 4); `--min-reads` and `--dry-run` supported.
  - `attribute/launch.py` — Submits SLURM jobs for attributions, one per (experiment, head) pair. Skips experiments with incomplete training (any fold model missing) or existing output; `--min-reads` filter defaults to 0 (disabled). `--head` is repeatable (default: `profile`). Logs to `logs/bpnet_attr/`.
  - `attribute/attribution_to_bigwig.py` — Converts hypothetical attribution NPZ files plus OHE NPZ files into observed-nucleotide BigWigs. Uses filtered peaks by default and averages overlapping attribution windows base-by-base.
  - `attribute/launch_bigwig_conversion.py` — Submits attribution-to-BigWig SLURM jobs, one per (experiment, head) pair. Skips missing attribution/OHE files and existing BigWigs. Logs to `logs/bpnet_attr_bigwig/`.
  - `modisco/launch.py` — Submits SLURM jobs to run `modisco motifs` on BPNet attributions, one job per (experiment, head) pair. Skips experiments where the attribution or OHE npz is missing, or the `.h5` output already exists; `--min-reads` filter supported (default: 0, disabled). Modisco parameters (`-n/--n-seqlets`, `-l/--leiden`, `-w/--window`) and SLURM resources are configurable. Output: `modisco/bpnet/{exp_id}_{head}.modisco.h5`. Logs to `logs/bpnet_modisco/`. Run `launch_report.py` separately after this completes.
  - `modisco/launch_report.py` — Submits SLURM jobs to run `modisco report` on completed `.h5` outputs, one job per (experiment, head) pair. Skips pairs where the `.h5` is missing or the report directory already exists. Uses JASPAR 2026 vertebrate motif database at `data/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt` for motif matching. Output: `modisco/bpnet/{exp_id}_{head}.modisco/`. Logs to `logs/bpnet_modisco/`.
  - `modisco/relaunch_timeout.py` — Recovers from SLURM time-limit cancellations. Scans `logs/bpnet_modisco/*.err` for TIME_LIMIT/TIMEOUT signals and resubmits the affected jobs with a longer runtime (default: `6-23:00:00`, vs the original `2-00:00:00`). Skips jobs whose `.h5` output already exists (completed on a prior retry). `--list` prints timed-out jobs and their status without submitting; `--dry-run` previews sbatch scripts.
  - `modisco/modisco.sh` — Manual SLURM array script (array indices 0–1 → count/profile) for running `modisco motifs` + `modisco report` on a single hardcoded experiment. Template for one-off runs; edit `biosample` and `background` variables before use.
  - `modisco/modisco_peak_test.sh` — Manual script for cross-experiment modisco: runs `modisco motifs` + `modisco report` using attributions from one model applied to peaks from a different experiment. Useful for testing how a model's learned motifs generalize to a different peak set.
  - `modisco/modisco_retrain.sh` — Similar to `modisco_peak_test.sh` but for retrained models (model name encodes both the track source and peak source). Used for benchmarking retrained models against alternative peak sets.
- **`src/cherimoya/`** — Cherimoya deep learning model training and evaluation (similar API to BPNet):
  - `fit/fit_cherimoya.py` — Production Cherimoya training script. Muon + AdamW dual-optimizer setup with warmup + cosine decay schedules. Accepts the same `--background NAME:RATIO` pattern as BPNet plus model/training controls such as `--n-filters`, `--n-layers`, `--batch-size`, `--max-epochs`, and `--early-stopping`. Output directory defaults to `models/cherimoya/{experiment}`; non-default backgrounds append a suffix.
  - `fit/muon.py` — Local Muon optimizer fallback for PyTorch versions without `torch.optim.Muon`.
  - `fit/test_muon.py` — Equivalence tests comparing the local Muon fallback to upstream `torch.optim.Muon` where available.
  - `fit/data_loader.py` — `PeakGenerator` and `PeakNegativeSampler` for strand-specific PRO-cap data loading during Cherimoya training.
  - `fit/launch.py` — Submits SLURM jobs to train Cherimoya models through an Apptainer image, one per (experiment, fold) pair. Skips experiments with fewer than `--min-reads` total reads (default: 0) and folds where the model file already exists. Extra arguments forwarded via `--fit-args`. Logs to `logs/cherimoya_fit/`.
  - `benchmark/benchmark_cherimoya.py` — Evaluates a trained Cherimoya across all folds on held-out test chromosomes. Defaults to `models/cherimoya/{experiment}`; use `--model-dir` to override. Reports per-fold and genome-wide profile Pearson, profile JSD, log-counts Pearson, and counts Spearman. Always writes results to `performance_metrics/cherimoya/{model_dir_name}.json`. `--save-output` writes scaled predictions to `predictions/cherimoya/{model_dir_name}.npz`.
  - `benchmark/consolidate_metrics.py` — Consolidates per-experiment Cherimoya benchmark JSON files into `performance_metrics/cherimoya/procap-atlas_performance_metrics.tsv`.
  - `n_filters/launch.py` — Submits Cherimoya `n_filters` sweep training jobs for values `16, 24, 36, 48, 64, 96, 196, 256` into `models/cherimoya_n_filters/`.
  - `n_filters/launch_benchmark.py` — Benchmarks completed `n_filters` sweep models.
  - `n_filters/consolidate_metrics.py` — Consolidates and plots `n_filters` sweep metrics in `performance_metrics/cherimoya_n_filters/`.
- **`src/hub/`** — UCSC track hub generation (see `src/hub/README.md` for full details):
  - `generate_hub.py` — Reads `configs/experiment_config.yaml` and writes `hub/hub.txt`, `hub/genomes.txt`, and `hub/hg38/trackDb.txt`. Experiments grouped into supertracks by biosample; each experiment gets a `compositeTrack` for plus/minus strand BigWigs, a `bigBed` peaks track, and optional attribution dynseq tracks. Experiments with `uncapped` in `library_construction` are set to `visibility hide`. Requires `--email`; `--base-url` sets the public-facing URL root (default: `https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas`).
  - `convert_peaks_bigbed.py` — Converts all `data/processed/peaks/{exp}_{biosample}.bed.gz` files to bigBed format using `bedToBigBed`. Writes alongside source `.bed.gz` files by default; use `--output-dir` to override. Handles 8-column merged, 6-column bidirectional-only, and other PINTS formats automatically. Skips experiments where the output already exists or input is missing. Supports `-j/--jobs` for parallelism.
- **`src/analysis/`** — Atlas-level analyses, currently count-correlation clustermaps at union peaks with optional BPNet/Cherimoya predictions.
- **`src/metaplot/`** — TSS-centered PRO-cap metaplots and heatmaps from GENCODE annotations.
- **`src/metaformer/`** — PromoterAI/MetaFormer helper scripts: config-to-BigWig TSV conversion, chromosome preprocessing, and cluster-specific training launchers.
- **`src/procapnet/`** — ProCapNet benchmark helper.
- **`tests/`** — Unit tests, currently focused on attribution BigWig conversion helpers.
- **`data/`** (gitignored, created by scripts) — Downloaded genome, signal files, and peaks.

## Environment Setup

Requires Python 3.12. Command line and Python dependencies are specified in `environment.yml`:

```bash
mamba env create -f environment.yml
mamba activate procap-atlas
```

## Pipeline Commands

Scripts must run in order: download → generate config → merge/rename → gc negatives → model training.

### Data Download

```bash
bash src/download/download_genome.sh      # GRCh38 reference -> data/hg38.fa, .fai, .chrom.sizes
bash src/download/download_bigwigs.sh     # Plus/minus strand BigWigs (8 parallel jobs) -> data/raw/bigwigs/
bash src/download/download_peaks.sh       # Peak BED files (4 parallel jobs) -> data/raw/peaks/
bash src/download/download_annotations.sh # GENCODE v49 annotation -> data/gencode.v49.annotation.gff3.gz
```

### Preprocessing

Preprocessing scripts are typically run from the repo root, but can be run from anywhere:

```bash
python src/preprocess/generate_config.py   # Manifest → configs/experiment_config.yaml
python src/preprocess/merge_bigwigs.py     # Merge replicate BigWigs (-j/--jobs flag, default 4 workers)
python src/preprocess/process_peaks.py     # Merge bidirectional and unidirectional peak files
python src/preprocess/filter_peaks_run.py  # Filter peaks containing non-ACGT sequence (-j/--jobs, default 4)
python src/preprocess/make_union_peaks.py  # Build data/processed/peaks/union_peaks.bed.gz
python src/preprocess/gc_match_run.py      # GC-matched negatives (-j/--jobs flag, default 1 worker)
python src/preprocess/count_reads.py       # Count reads per experiment → configs/n_reads.txt
```

### Model Training

```bash
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0                                          # Train BPNet, default GC background (1/7)
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background gc:0.1               # GC negatives only
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background ccre:0.05 --background gc:0.05  # custom ratios
python src/bpnet/fit/launch.py                     # submit all experiments x 7 folds via SLURM
python src/bpnet/fit/launch.py --dry-run           # preview sbatch scripts without submitting
python src/bpnet/model_upload.py                   # upload BPNet models/configs/config.json to Hugging Face
```

### Cherimoya Training

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0                   # train Cherimoya, default backgrounds
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1  # GC negatives only
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --n-filters 196 --batch-size 32
python src/cherimoya/fit/launch.py                                                    # submit all experiments x 7 folds via SLURM
python src/cherimoya/fit/launch.py --dry-run                                          # preview sbatch scripts without submitting
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM                 # benchmark across all folds
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output   # also save predictions
python src/cherimoya/benchmark/consolidate_metrics.py                                # consolidate benchmark JSON to TSV
python src/cherimoya/n_filters/launch.py --dry-run                                   # preview n_filters sweep training
python src/cherimoya/n_filters/launch_benchmark.py --dry-run                         # preview n_filters sweep benchmarks
python src/cherimoya/n_filters/consolidate_metrics.py                                # consolidate/plot n_filters metrics
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
python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM          # convert observed attribution scores to BigWig
python src/bpnet/attribute/launch_bigwig_conversion.py --dry-run            # preview attribution BigWig conversion jobs
```

### Modisco

```bash
# Step 1: run modisco motifs (requires attributions from attribute/launch.py)
python src/bpnet/modisco/launch.py                                              # submit all experiments, both heads
python src/bpnet/modisco/launch.py --head profile --head count                  # explicit both heads
python src/bpnet/modisco/launch.py --head count                                 # count head only
python src/bpnet/modisco/launch.py --dry-run                                    # preview without submitting
python src/bpnet/modisco/launch.py --min-reads 20000000                         # only well-covered experiments
python src/bpnet/modisco/launch.py -n 500000 -l 30 -w 500                      # custom modisco parameters

# Step 2: run modisco report (requires completed .h5 outputs from step 1)
python src/bpnet/modisco/launch_report.py                                       # submit all experiments, both heads
python src/bpnet/modisco/launch_report.py --head profile                        # profile head only
python src/bpnet/modisco/launch_report.py --dry-run                             # preview without submitting

# Recovery: relaunch jobs that hit the SLURM time limit
python src/bpnet/modisco/relaunch_timeout.py --list                             # show timed-out jobs and status
python src/bpnet/modisco/relaunch_timeout.py --dry-run                          # preview relaunch scripts
python src/bpnet/modisco/relaunch_timeout.py                                    # resubmit with 6-23:00:00 time limit
python src/bpnet/modisco/relaunch_timeout.py --time 4-00:00:00                  # custom extended time limit
```

### Track Hub

Requires processed BigWig and peak files. Run after preprocessing is complete.

```bash
python src/hub/generate_hub.py --email you@example.com  # write hub/hub.txt, genomes.txt, trackDb.txt
python src/hub/generate_hub.py --email you@example.com --no-attributions
python src/hub/convert_peaks_bigbed.py                  # convert peaks to bigBed → hub/hg38/bigbed/
python src/hub/convert_peaks_bigbed.py -j 4             # 4 parallel workers
hubCheck hub/hub.txt                                    # validate (requires UCSC Kent tools)
```

Hub URL: `https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas/hub/hub.txt`

## External Dependencies

- `wget`, `gunzip` — file downloading and decompression
- `samtools` — FASTA indexing
- `bgzip` (htslib) — bgzip compression of processed peak and negative BED files
- `parallel` (GNU parallel) — parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` (UCSC Kent tools) — BigWig merging
- `bedToBigBed` (UCSC Kent tools, optional) — peak BED to bigBed conversion for track hub
- `hubCheck` (UCSC Kent tools, optional) — track hub validation
- `sbatch`, `apptainer` (SLURM/container runtime, optional) — cluster job submission for model training, attribution, modisco, and sweep launchers
- Python (see `environment.yml`): `pyyaml`, `pybigtools`, `cherimoya`, `bpnet-lite`, plus conda-provided tools including `matplotlib`

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
    ├── peaks/      # merged, filtered, and union peak BED.gz files
    └── negatives/  # GC-matched negative BED.gz files
```
