# PRO-cap atlas

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97-Weights-yellow)](https://huggingface.co/collections/adamyhe/procap-atlas)

Preprocessing, modeling, and atlas-level analysis for ENCODE PRO-cap data. PRO-cap (Precision Run-On sequencing with cap selection) profiles transcription start sites and promoter-proximal initiation.

This repository contains the data preparation pipeline, trained-model workflows,
benchmarking scripts, attribution and motif analysis utilities, and UCSC track
hub generation code used for the atlas.

## Status

- **BPNet** is the primary deployed model family. Training, benchmarking,
  attribution, MoDISco, motif clustering, Hugging Face upload, and track hub
  support are documented in [`src/bpnet/`](src/bpnet/README.md).
- **Cherimoya** is a novel architecture developed by Jacob Schreiber.
- Cherimoya PRO-cap models are still in development (currently waiting on stable-ish
  API and DeepLIFT/SHAP support).
- **MetaFormer / PromoterAI** support is still in development. The scripts in
  [`src/metaformer/`](src/metaformer/README.md) are experimental helpers and
  cluster templates.
- **ProCapNet** code is retained for legacy model benchmarking only. See
  [`src/procapnet/`](src/procapnet/README.md).

## Environment

Python and command-line dependencies are specified in `environment.yml`:

```bash
mamba env create -f environment.yml
mamba activate procap-atlas
```

Some workflows also require external tools:

- `wget`, `parallel`, `samtools`, `bgzip`
- UCSC Kent tools: `bigWigMerge`, `bedGraphToBigWig`, `bedToBigBed`, `hubCheck`
- Optional cluster tools: `sbatch`, `apptainer`

MotifCompendium clustering uses a separate environment. See
[`src/bpnet/README.md`](src/bpnet/README.md#motif-clustering).

## Quickstart

Run the main atlas pipeline from the repository root:

```bash
# 1. Download raw inputs
bash src/download/download_genome.sh
bash src/download/download_bigwigs.sh
bash src/download/download_peaks.sh
bash src/download/download_annotations.sh

# 2. Build the experiment config and processed tracks
python src/preprocess/generate_config.py
python src/preprocess/merge_bigwigs.py
python src/preprocess/process_peaks.py
python src/preprocess/filter_peaks_run.py
python src/preprocess/make_union_peaks.py
python src/preprocess/gc_match_run.py
python src/preprocess/count_reads.py

# 3. Train and evaluate BPNet
python src/bpnet/fit/fit_bpnet.py -e ENCSR882DWM --fold 0
python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM

# 4. Generate model attributions and motif analyses
python src/bpnet/attribute/save_ohe.py -e ENCSR882DWM
python src/bpnet/attribute/attribute_bpnet.py -e ENCSR882DWM
python src/bpnet/modisco/launch.py --dry-run

# 5. Build browser tracks
python src/hub/convert_peaks_bigbed.py --dry-run
python src/hub/generate_hub.py --email you@example.com
```

Most commands support additional options for parallelism, read-count filters,
custom model directories, dry runs, and SLURM submission. The directory READMEs
below are the source of truth for those details.

All SLURM launch scripts in this repository use hard-coded defaults for the
Sherlock HPC environment. Apptainer usage is also a Sherlock-specific workaround,
not a general requirement for every cluster.

## Workflow Guides

- [`src/download/`](src/download/README.md): reference genome, ENCODE BigWig,
  peak BED, and GENCODE downloads.
- [`src/preprocess/`](src/preprocess/README.md): manifest parsing, BigWig
  merging, peak merging/filtering, union peaks, GC negatives, and read counts.
- [`src/bpnet/`](src/bpnet/README.md): BPNet training, benchmarking,
  attributions, MoDISco, motif clustering, uploads, and model track conversion.
- [`src/cherimoya/`](src/cherimoya/README.md): Cherimoya training,
  benchmarking, and architecture sweeps (IN DEVELOPMENT).
- [`src/hub/`](src/hub/README.md): UCSC track hub generation, bigBed
  conversion, Hugging Face track hosting, and validation.
- [`src/analysis/`](src/analysis/README.md): atlas-level count correlations and
  model warning flags.
- [`src/metaplot/`](src/metaplot/README.md): TSS-centered PRO-cap metaplots and
  heatmaps.
- [`src/metaformer/`](src/metaformer/README.md): PromoterAI /
  MetaFormer helpers (IN DEVELOPMENT).
- [`src/procapnet/`](src/procapnet/README.md): legacy ProCapNet benchmarking.

## Repository Layout

```text
data_manifests/   ENCODE URL manifests, experiment metadata, archive blacklist
configs/          Generated experiment config, chromosome splits, read counts
config.json       Hugging Face BPNet model metadata
src/download/     Download scripts for genome, signals, peaks, annotations
src/preprocess/   Processing pipeline for experiment-level model inputs
src/bpnet/        Primary BPNet model training, evaluation, attribution, motifs
src/cherimoya/    Cherimoya model training and evaluation, not deployed yet
src/hub/          UCSC track hub and track asset utilities
src/analysis/     Atlas-level correlation and warning-flag analyses
src/metaplot/     TSS-centered signal plots
src/metaformer/   Experimental PromoterAI / MetaFormer helpers
src/procapnet/    Legacy ProCapNet benchmark helper
tests/            Unit tests, currently focused on attribution BigWig helpers
data/             Gitignored downloaded and processed data
models/           Gitignored trained model artifacts
```

## Data Layout

After download and preprocessing, the working data directory is organized as:

```text
data/
+-- hg38.fa
+-- hg38.fa.fai
+-- hg38.chrom.sizes
+-- raw/
|   +-- bigwigs/
|   +-- peaks/
+-- processed/
    +-- bigwigs/
    +-- peaks/
    +-- negatives/
```

The public UCSC track hub is:

```text
https://mitra.stanford.edu/kundaje/oak/ayhe/procap-atlas/hub/hub.txt
```
