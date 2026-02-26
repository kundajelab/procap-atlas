# PRO-cap atlas

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Prerequisites

- `wget`, `gunzip` -- file downloading and decompression
- `samtools` -- FASTA indexing
- `bgzip` (htslib) -- bgzip compression of processed BED files
- GNU `parallel` -- parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` ([UCSC Kent tools](https://hgdownload.soe.ucsc.edu/admin/exe/)) -- BigWig merging
- `sbatch` (SLURM, optional) -- cluster job submission via `src/bpnet/fit/launch.py` and `src/bpnet/attribute/launch.py`
- Python 3.10+ (`pip install -r requirements.txt`)

## Sample installation with mamba

```bash
# Create and activate mamba environment
mamba create -n procap-atlas python=3.12 -y
mamba activate procap-atlas

# Install command-line tools
mamba install -c bioconda -c conda-forge samtools ucsc-bigwigmerge ucsc-bedgraphtobigwig -y
# Usually not needed.
# mamba install -c conda-forge parallel wget -y

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download data

Download scripts can be run from anywhere in the project directory, but we run them from root when possible for consistency:

```bash
# Download and index GRCh38 reference genome
bash src/download/download_genome.sh

# Download plus/minus strand BigWig files (8 parallel jobs)
bash src/download/download_bigwigs.sh

# Download bidirectional peak BED files (4 parallel jobs)
bash src/download/download_peaks.sh
```

### 2. Generate experiment config

Cross-references the experiment report TSV with manifest URL files to produce a YAML config. Archived file IDs listed in `data_manifests/archive_blacklist.txt` are excluded. When bidirectional peaks are missing for an experiment, divergent peaks are used as a fallback (with a warning).

```bash
python src/preprocess/generate_config.py
```

Output: `configs/experiment_config.yaml`

### 3. Merge replicate BigWig files

Merges replicate BigWig files per experiment, keeping plus and minus strands separate. Single-replicate experiments are copied without reprocessing.

```bash
python src/preprocess/merge_bigwigs.py
```

Output: `data/processed/bigwigs/{experiment}_{biosample}_{strand}.bigWig`

### 4. Merge and process peak files

Merge bidirectional and unidirectional peak BED files per experiment into a single processed file. Falls back to copying when only one peak type is available.

```bash
python src/preprocess/process_peaks.py
```

Output: `data/processed/peaks/{experiment}_{biosample}.bed.gz`

### 5. Generate GC-matched negatives

For each experiment, generates GC-matched negative regions from the genome, optionally filtering by signal strength using both strand BigWigs. Output is bgzip-compressed BED.

```bash
python src/preprocess/gc_match_run.py          # default 1 worker
python src/preprocess/gc_match_run.py -j 8     # 8 parallel workers
```

Output: `data/processed/negatives/{experiment}_{biosample}_{peak_type}_gc_negatives.bed.gz`

### 6. Count reads per experiment

Counts total reads across both strand BigWigs for each experiment using `pybigtools`. Results are written as a TSV.

```bash
python src/preprocess/count_reads.py
```

Output: `configs/n_reads.txt`

### 7. Train BPNet model

Trains a BPNet model for a single experiment using 7-fold chromosome cross-validation. Fold `i` is held out for testing, fold `(i+1) % 7` for validation, and the remaining 5 folds for training. Chromosome splits are defined in `configs/chrom_splits.yaml`.

Background negative sources and their per-batch sampling ratios are specified with repeatable `--background NAME:RATIO` arguments (available names: `ccre`, `gc`). The output directory name encodes the background configuration.

```bash
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background ccre:0.05 --background gc:0.05
```

Output: `models/bpnet/{experiment}_{background_config}/`

To submit training jobs for all experiments and folds via SLURM, use `launch.py`. Jobs are skipped automatically for experiments with fewer than `--min-reads` total reads (default: 10M) and for folds where the model file already exists. **NOTE** that the SLURM launcher has some hardcoded references to partitions/module loads that are specific to Sherlock/Kundaje lab, so you will have to tweak this if you want to retrain everything on your own SLURM cluster.

```bash
python src/bpnet/fit/launch.py                          # all experiments x 7 folds
python src/bpnet/fit/launch.py --dry-run                # preview without submitting
python src/bpnet/fit/launch.py --min-reads 20000000     # only well-covered experiments
```

### 8. Benchmark BPNet model

Evaluates trained models across all folds on held-out test chromosomes. Use the same `--background` arguments as during training to resolve the model directory, or specify `--model-dir` directly.

```bash
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --background gc:0.1
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_dnase
```

Output: `performance_metrics/bpnet/{model_dir_name}.json`

### 9. Compute attributions

Computes DeepLIFT/SHAP attributions across all folds, averaged genome-wide. Use the same `--background` arguments as during training, or `--model-dir` directly. `--head` selects the profile or count output head; `--ohe` additionally saves one-hot-encoded sequences.

```bash
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head count --ohe
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_dnase
```

Output: `attributions/bpnet/{model_dir_name}_{head}.npz`

To submit attribution jobs for all experiments via SLURM, use `launch.py`. Jobs are skipped automatically if the output already exists, any fold model is missing, or the experiment has fewer than `--min-reads` total reads (default: 10M). `--head` is repeatable to run multiple heads. **NOTE** that the SLURM launcher has some hardcoded references to partitions/module loads that are specific to Sherlock/Kundaje lab, so you will have to tweak this if you want to use this to run attributions on your own SLURM cluster.

```bash
python src/bpnet/attribute/launch.py                              # profile head, all experiments
python src/bpnet/attribute/launch.py --head profile --head count  # both heads
python src/bpnet/attribute/launch.py --dry-run                    # preview without submitting
```

## Project structure

```
data_manifests/          # ENCODE file URLs, experiment metadata, archive blacklist
configs/                 # Generated YAML experiment config, chromosome fold splits.
src/
  download/              # Bash download scripts
  preprocess/            # Python preprocessing scripts
    generate_config.py   # Manifest parsing and config generation
    merge_bigwigs.py     # Replicate BigWig merging
    process_peaks.py     # Merge bidirectional + unidirectional peaks
    gc_match.py          # GC-matched negative region extraction (adapted from tangermeme)
    gc_match_run.py      # Runs gc_match for all experiments in config
    count_reads.py       # Count total reads per experiment across both strand BigWigs
  bpnet/                 # BPNet deep learning model
    fit/fit_bpnet.py           # BPNet training script with configurable background sampling
    fit/launch.py              # SLURM job submission for training
    benchmark/benchmark_bpnet.py     # BPNet evaluation across folds → performance_metrics/
    attribute/attribute_bpnet.py     # DeepLIFT/SHAP attributions → attributions/
    attribute/launch.py              # SLURM job submission for attributions
  alphagenome/           # AlphaGenome analysis (planned)
data/                    # gitignored, created by scripts
  hg38.fa                # Reference genome + index
  raw/                   # Downloaded files (per replicate)
    bigwigs/
    peaks/
  processed/             # Merged/renamed files (per experiment)
    bigwigs/
    peaks/
    negatives/
```
