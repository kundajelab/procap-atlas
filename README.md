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

Scripts in `src/bpnet/motifcompendium/` should be run inside this environment after modisco outputs have been generated (step 12).

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

# Download GENCODE v49 annotation for TSS metaplots
bash src/download/download_annotations.sh
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

### 5. Filter peaks and build union peaks

Filters processed peak files to remove loci whose model input window contains non-ACGT sequence, then optionally builds a merged non-overlapping peak set across experiments for atlas-level analyses.

```bash
python src/preprocess/filter_peaks_run.py          # default 4 workers
python src/preprocess/filter_peaks_run.py -j 8     # 8 parallel workers
python src/preprocess/make_union_peaks.py          # all experiments
python src/preprocess/make_union_peaks.py --min-reads 10000000 --slop 100
python src/preprocess/make_union_peaks.py --dry-run
```

Outputs: `data/processed/peaks/{experiment}_{biosample}_filtered.bed.gz`, `data/processed/peaks/union_peaks.bed.gz`

### 6. Generate GC-matched negatives

For each experiment, generates GC-matched negative regions from the genome, optionally filtering by signal strength using both strand BigWigs. Output is bgzip-compressed BED.

```bash
python src/preprocess/gc_match_run.py          # default 1 worker
python src/preprocess/gc_match_run.py -j 8     # 8 parallel workers
```

Output: `data/processed/negatives/{experiment}_{biosample}_gc_negatives.bed.gz`

### 7. Count reads per experiment

Counts total reads across both strand BigWigs for each experiment using `pybigtools`. Results are written as a TSV.

```bash
python src/preprocess/count_reads.py
```

Output: `configs/n_reads.txt`

### 8. Train BPNet model

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

### 9. Benchmark BPNet model

Evaluates trained models across all folds on held-out test chromosomes. Use `--model-dir` for any non-default model directory.

```bash
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_dnase
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --save-output
```

Output: `performance_metrics/bpnet/{model_dir_name}.json`; with `--save-output`, `predictions/bpnet/{experiment}.npz`

### 10. Upload BPNet models to Hugging Face

The repository includes a top-level `config.json` with BPNet atlas metadata for the Hugging Face model repo. Hugging Face uses root-level config files such as `config.json` as default query files for model download statistics, so keep this file in the model repo root when uploading.

```bash
python src/bpnet/model_upload.py
```

This uploads the BPNet model folder, `configs/`, and root `config.json` to `adamyhe/procap-atlas`.

### 11. Compute attributions

Computes DeepLIFT/SHAP attributions across all folds, averaged genome-wide, using `processed.filtered_peaks` from step 5. Custom model directories can be loaded using `--model-dir` if not using defaults. `--head` selects the profile or count output head. One-hot-encoded sequences must be saved separately with `save_ohe.py` before running modisco (step 12).

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

Attribution NPZ files can be converted to observed-nucleotide BigWigs for UCSC display. The converter multiplies hypothetical attributions by matching OHE sequence and averages overlaps base-by-base.

```bash
python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM
python src/bpnet/attribute/attribution_to_bigwig.py -e ENCSR882DWM --head count
python src/bpnet/attribute/launch_bigwig_conversion.py --dry-run
python src/bpnet/attribute/launch_bigwig_conversion.py --head profile --head count
```

Output: `attributions/bpnet/bigwigs/{experiment}_{head}.bigWig`

### 12. Run tf-modisco

Discovers sequence motifs from DeepLIFT/SHAP attributions using [tf-MoDISco](https://github.com/jmschrei/tfmodisco-lite). Requires both the attribution `.npz` and OHE `.npz` files from step 11.

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

### 13. Cluster motifs across experiments

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
cluster report HTML/PDF include forward and reverse logos and are sorted by
`total_seqlets`, the summed number of seqlets per final motif cluster.

### 14. Train Cherimoya model

Trains a [Cherimoya](https://github.com/jmschrei/cherimoya) model for a single experiment using the same 7-fold chromosome cross-validation scheme as BPNet. Training uses a dual-optimizer setup: Muon for 2D weight matrices and AdamW for the remaining parameters, both with warmup + cosine decay schedules. The script accepts `--n-filters`, `--n-layers`, `--batch-size`, `--max-epochs`, `--early-stopping`, and related training controls.

Background negative sources follow the same `--background NAME:RATIO` interface as BPNet.

```bash
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --background gc:0.1
python src/cherimoya/fit/fit_cherimoya.py -e ENCSR882DWM --fold 0 --n-filters 196 --batch-size 32
```

Output: `models/cherimoya/{experiment}/` (default) / `models/cherimoya/{experiment}_{background_config}/`

To submit training jobs for all experiments and folds via SLURM:

```bash
python src/cherimoya/fit/launch.py             # all experiments x 7 folds
python src/cherimoya/fit/launch.py --dry-run   # preview without submitting
python src/cherimoya/fit/launch.py --min-reads 20000000 --fit-args "--max-epochs 100"
```

The SLURM launcher runs training through an Apptainer image by default (`/scratch/users/ayhe/apptainer/cherimoya.sif`); override it with `--apptainer-image` and `--apptainer-bind` for another cluster.

### 15. Benchmark Cherimoya model

Evaluates trained Cherimoya models across all folds on held-out test chromosomes. Optionally saves scaled predictions.

```bash
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --model-dir models/cherimoya/ENCSR882DWM_gc0.1
python src/cherimoya/benchmark/benchmark_cherimoya.py -e ENCSR882DWM --save-output
```

Output: `performance_metrics/cherimoya/{model_dir_name}.json`

To consolidate per-experiment Cherimoya metrics into a TSV:

```bash
python src/cherimoya/benchmark/consolidate_metrics.py
```

Output: `performance_metrics/cherimoya/procap-atlas_performance_metrics.tsv`

### 16. Cherimoya n_filters sweep

Launches a sweep over `n_filters` values (`16, 24, 36, 48, 64, 96, 196, 256`), benchmarks the trained sweep models, and consolidates metrics plus plots.

```bash
python src/cherimoya/n_filters/launch.py --dry-run
python src/cherimoya/n_filters/launch.py --min-reads 20000000
python src/cherimoya/n_filters/launch_benchmark.py --dry-run
python src/cherimoya/n_filters/launch_benchmark.py --min-reads 20000000
python src/cherimoya/n_filters/consolidate_metrics.py
```

Outputs: `models/cherimoya_n_filters/{experiment}_nf{n_filters}/`, `performance_metrics/cherimoya/{experiment}_nf{n_filters}.json`, `performance_metrics/cherimoya_n_filters/`

### Optional: MetaFormer / PromoterAI

The MetaFormer helpers convert the PRO-cap config into a PromoterAI BigWig target TSV, preprocess chromosome-sharded HDF5 inputs, and launch multi-GPU training. The SLURM scripts are cluster-specific templates.

```bash
python src/metaformer/procap_config_to_promoterai.py
python src/metaformer/procap_config_to_promoterai.py --absolute-paths --require-files
sbatch src/metaformer/preprocess.sh
sbatch src/metaformer/train.sh
sbatch src/metaformer/train_bridges2.sh
```

Output: `configs/promoterai_procap_bigwigs.tsv`, `data/promoterai/`, `models/metaformer/all_tracks/`

### Optional: Atlas analyses

Additional analysis scripts generate count-correlation clustermaps at union peaks and TSS-centered signal metaplots/heatmaps.

```bash
python src/analysis/count_correlation.py
python src/analysis/count_correlation.py --model bpnet --min-reads 10000000 --device cuda
python src/metaplot/metaplot_tss.py --plot-type both --min-reads 10000000
python src/metaplot/metaplot_tss.py --experiment ENCSR882DWM --feature transcript
```

Outputs: `figures/count_correlation/`, `figures/metaplots/`

### Optional: Generate UCSC track hub

Generates a [UCSC track hub](https://genome.ucsc.edu/goldenPath/help/hgTrackHubHelp.html) for visualizing all experiments in the Genome Browser. Experiments are grouped by biosample; each has a strand-specific multiWig signal track, peaks track, and optional attribution dynseq tracks.

```bash
python src/hub/generate_hub.py --email you@example.com  # write hub files
python src/hub/generate_hub.py --email you@example.com --no-attributions
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
config.json              # Hugging Face BPNet model metadata and download tracking query file
src/
  download/              # Bash download scripts
  preprocess/            # Python preprocessing scripts
    generate_config.py   # Manifest parsing and config generation
    merge_bigwigs.py     # Replicate BigWig merging
    process_peaks.py     # Merge bidirectional + unidirectional peaks
    _gc_match.py         # GC-matched negative region extraction (adapted from tangermeme)
    gc_match_run.py      # Runs _gc_match for all experiments in config
    filter_peaks_run.py  # Remove peaks with non-ACGT sequence in the model input window
    make_union_peaks.py  # Build a merged atlas-wide union peak set
    count_reads.py       # Count total reads per experiment across both strand BigWigs
  hub/                   # UCSC track hub generation (see src/hub/README.md)
    generate_hub.py      # Generate hub.txt, genomes.txt, trackDb.txt from experiment config
    convert_peaks_bigbed.py  # Convert peak BED.gz files to bigBed format
  analysis/              # Atlas-level count-correlation analyses
  metaplot/              # TSS-centered signal metaplots and heatmaps
  metaformer/            # PromoterAI/MetaFormer TSV conversion, preprocessing, training scripts
  bpnet/                 # BPNet deep learning model
    model_upload.py            # Upload BPNet artifacts, configs, and config.json to Hugging Face
    fit/fit_bpnet.py           # BPNet training script with configurable background sampling
    fit/launch.py              # SLURM job submission for training
    benchmark/benchmark_bpnet.py     # BPNet evaluation across folds → performance_metrics/
    attribute/attribute_bpnet.py     # DeepLIFT/SHAP attributions → attributions/
    attribute/save_ohe.py            # Save one-hot-encoded sequences → attributions/
    attribute/run_ohe.py             # Save OHE for all experiments (async, -j concurrency)
    attribute/launch.py              # SLURM job submission for attributions
    attribute/attribution_to_bigwig.py       # Convert observed attribution scores to BigWig
    attribute/launch_bigwig_conversion.py    # SLURM launcher for attribution BigWig conversion
    modisco/launch.py                # SLURM job submission for modisco motifs
    modisco/launch_report.py         # SLURM job submission for modisco report
    modisco/relaunch_timeout.py      # Relaunch SLURM jobs that hit the time limit
    motifcompendium/cluster_motifs.py  # Cross-experiment motif clustering via MotifCompendium
  cherimoya/             # Cherimoya deep learning model
    fit/fit_cherimoya.py       # Production training script
    fit/muon.py                # Local Muon optimizer fallback for PyTorch versions without torch.optim.Muon
    fit/test_muon.py           # Equivalence tests for local Muon fallback
    fit/data_loader.py         # Strand-specific PRO-cap data loader
    fit/launch.py              # SLURM job submission for training
    benchmark/benchmark_cherimoya.py # Cherimoya evaluation across folds → performance_metrics/
    benchmark/consolidate_metrics.py # Consolidate benchmark JSON files into a TSV
    n_filters/                 # Cherimoya n_filters sweep launch, benchmark, and plots
  procapnet/             # ProCapNet benchmark helper
tests/                   # Unit tests for attribution BigWig conversion helpers
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
