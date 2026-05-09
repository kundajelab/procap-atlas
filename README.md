# PRO-cap atlas

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Environment installation

Command line and python dependencies are specified in `environment.yml` and can be installed via conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate procap-atlas
```

`parallel` and `wget` are usually installed by default, but if not, they can also be installed via conda/mamba.

### MotifCompendium environment

[MotifCompendium](https://github.com/kundajelab/MotifCompendium) requires a separate conda environment. Clone the repo and create the environment (GPU or CPU variant):

```bash
git clone https://github.com/kundajelab/MotifCompendium.git
cd MotifCompendium

# GPU (recommended)
conda env create -f environment_gpu.yml
conda activate motifcompendium-gpu

# CPU-only
conda env create -f environment.yml
conda activate motifcompendium

pip install -e .
```

Scripts in `src/bpnet/motifcompendium/` should be run inside this environment after modisco outputs have been generated (step 10).

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
python src/preprocess/merge_bigwigs.py        # default 4 workers
python src/preprocess/merge_bigwigs.py -j 8   # 8 parallel workers
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

Background negative sources and their per-batch sampling ratios are specified with repeatable `--background NAME:RATIO` arguments (available names: `ccre`, `gc`). The output directory name encodes the background configuration. By default, a ratio of 7:1 peaks:GC-matched backgrounds are used in training.

```bash
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0 --background ccre:0.05 --background gc:0.05
```

Output: `models/bpnet/{experiment}/` (default) / `models/bpnet/{experiment}_{background_config}/`

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

Computes DeepLIFT/SHAP attributions across all folds, averaged genome-wide. Custom model directories can be loaded using `--model-dir` if not using defaults. `--head` selects the profile or count output head. One-hot-encoded sequences must be saved separately with `save_ohe.py` before running modisco (step 10).

```bash
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --head count
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_dnase
python src/bpnet/attribute/save_ohe.py -e ENCSR882DWM
```

Output: `attributions/bpnet/{model_dir_name}_{head}.npz`, `attributions/bpnet/{experiment}_ohe.npz`

To submit attribution jobs for all experiments via SLURM, use `launch.py`. Jobs are skipped automatically if the output already exists, any fold model is missing, or the experiment has fewer than `--min-reads` total reads (default: 0, disabled). `--head` is repeatable to run multiple heads.

```bash
python src/bpnet/attribute/launch.py                              # profile head, all experiments
python src/bpnet/attribute/launch.py --head profile --head count  # both heads
python src/bpnet/attribute/launch.py --dry-run                    # preview without submitting
```

### 10. Run tf-modisco

Discovers sequence motifs from DeepLIFT/SHAP attributions using [tf-MoDISco](https://github.com/jmschrei/tfmodisco-lite). Requires both the attribution `.npz` and OHE `.npz` files from step 9.

Modisco runs in two stages: `modisco motifs` (compute-intensive, multi-CPU) then `modisco report` (lightweight, generates HTML report with JASPAR motif matches). These are submitted as separate SLURM jobs to allow the heavier motifs step to be restarted independently if it times out.

```bash
# Step 1: run modisco motifs
python src/bpnet/modisco/launch.py                               # both heads, all experiments
python src/bpnet/modisco/launch.py --head profile                # profile head only
python src/bpnet/modisco/launch.py --dry-run                     # preview without submitting
python src/bpnet/modisco/launch.py --min-reads 20000000          # only well-covered experiments

# Step 2: run modisco report (after step 1 completes)
python src/bpnet/modisco/launch_report.py                        # both heads, all experiments
python src/bpnet/modisco/launch_report.py --head profile         # profile head only
python src/bpnet/modisco/launch_report.py --dry-run              # preview without submitting

# Recovery: relaunch jobs that hit the SLURM time limit
python src/bpnet/modisco/relaunch_timeout.py --list              # show timed-out jobs and status
python src/bpnet/modisco/relaunch_timeout.py --dry-run           # preview relaunch scripts
python src/bpnet/modisco/relaunch_timeout.py                     # resubmit with 6-23:00:00 limit
python src/bpnet/modisco/relaunch_timeout.py --time 4-00:00:00   # custom extended time limit
```

Output: `modisco/bpnet/{exp_id}_{head}.modisco.h5`, `modisco/bpnet/{exp_id}_{head}.modisco/`

### 11. Cluster motifs across experiments

Clusters modisco motifs across all experiments using [MotifCompendium](https://github.com/kundajelab/MotifCompendium). Run inside the `motifcompendium` or `motifcompendium-gpu` conda environment (see installation above). The script expects modisco `.h5` files to be present in the working directory (or adjust glob paths in the script).

```bash
conda activate motifcompendium-gpu
python src/bpnet/motifcompendium/cluster_motifs.py
```

Output: `count_similarity_distribution.html`, `profile_similarity_distribution.html`

For an unfiltered MotifCompendium run that retains all clustered motifs, use the
separate all-motifs pipeline:

```bash
python src/bpnet/motifcompendium/cluster_motifs_all.py
```

Output: `motifcompendium/bpnet_all_motifs/`. The cluster metadata TSV and
cluster report HTML/PDF include `total_seqlets`, the summed number of seqlets per
final motif cluster.

### 12. Train Cherimoya model

Trains a [Cherimoya](https://github.com/jmschrei/cherimoya) model for a single experiment using the same 7-fold chromosome cross-validation scheme as BPNet. The production training script uses a Triton fused dilated conv+norm kernel (`CheriBlock`). Training uses a dual-optimizer setup: Muon for the bulk 2D weight matrices (linear layers in each block) and AdamW for everything else (input/output convolutions, biases, scalars), both with warmup + cosine decay schedules.

Background negative sources follow the same `--background NAME:RATIO` interface as BPNet.

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --muon-lr 0.005 --adam-lr 0.002
```

Output: `models/cherimoya/{experiment}/` (default) / `models/cherimoya/{experiment}_{background_config}/`

To submit training jobs for all experiments and folds via SLURM:

```bash
python src/cherimoya/fit/launch.py             # all experiments x 7 folds
python src/cherimoya/fit/launch.py --dry-run   # preview without submitting
```

### 13. Benchmark Cherimoya model

Evaluates trained Cherimoya models across all folds on held-out test chromosomes. Optionally saves scaled predictions.

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --model-dir models/cherimoya/ENCSR882DWM_gc0.1
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output
```

Output: `performance_metrics/cherimoya/{model_dir_name}.json`

### Optional: Generate UCSC track hub

Generates a [UCSC track hub](https://genome.ucsc.edu/goldenPath/help/hgTrackHubHelp.html) for visualizing all experiments in the Genome Browser. Experiments are grouped by biosample; each has a strand-specific multiWig signal track and a peaks track.

```bash
python src/hub/generate_hub.py --email you@example.com  # write hub files
python src/hub/convert_peaks_bigbed.py                  # convert peaks to bigBed (requires bedToBigBed)
python src/hub/convert_peaks_bigbed.py -j 4             # 4 parallel workers
hubCheck hub/hub.txt                                    # validate (optional)
```

Output: `hub/hub.txt`, `hub/genomes.txt`, `hub/hg38/trackDb.txt`, `hub/hg38/bigbed/*.bb`

Hub URL: `https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas/hub/hub.txt`

I will try to keep this hub online for as long as my accounts at Stanford remain active, but at some point my stuff will get deleted.

## Project structure

```
data_manifests/          # ENCODE file URLs, experiment metadata, archive blacklist
configs/                 # Generated YAML experiment config, chromosome fold splits
src/
  download/              # Bash download scripts
  preprocess/            # Python preprocessing scripts
    generate_config.py   # Manifest parsing and config generation
    merge_bigwigs.py     # Replicate BigWig merging
    process_peaks.py     # Merge bidirectional + unidirectional peaks
    gc_match.py          # GC-matched negative region extraction (adapted from tangermeme)
    gc_match_run.py      # Runs gc_match for all experiments in config
    count_reads.py       # Count total reads per experiment across both strand BigWigs
  hub/                   # UCSC track hub generation (see src/hub/README.md)
    generate_hub.py      # Generate hub.txt, genomes.txt, trackDb.txt from experiment config
    convert_peaks_bigbed.py  # Convert peak BED.gz files to bigBed format
  bpnet/                 # BPNet deep learning model
    fit/fit_bpnet.py           # BPNet training script with configurable background sampling
    fit/launch.py              # SLURM job submission for training
    benchmark/benchmark_bpnet.py     # BPNet evaluation across folds → performance_metrics/
    attribute/attribute_bpnet.py     # DeepLIFT/SHAP attributions → attributions/
    attribute/save_ohe.py            # Save one-hot-encoded sequences → attributions/
    attribute/run_ohe.py             # Save OHE for all experiments (async, -j concurrency)
    attribute/launch.py              # SLURM job submission for attributions
    modisco/launch.py                # SLURM job submission for modisco motifs
    modisco/launch_report.py         # SLURM job submission for modisco report
    modisco/relaunch_timeout.py      # Relaunch SLURM jobs that hit the time limit
    motifcompendium/cluster_motifs.py  # Cross-experiment motif clustering via MotifCompendium
  cherimoya/             # Cherimoya deep learning model
    fit/fit_cherimoya.py       # Production training script (Triton fused kernel)
    fit/fit_cherimoya2.py      # Testing/development training script (pure PyTorch)
    fit/data_loader.py         # Strand-specific PRO-cap data loader
    fit/launch.py              # SLURM job submission for training
    benchmark/benchmark_cherimoya.py # Cherimoya evaluation across folds → performance_metrics/
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
