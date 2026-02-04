# PRO-cap atlas

Preprocessing and deep learning-based analysis of the ENCODE PRO-cap atlas. PRO-cap (Precision Run-On sequencing with cap selection) data is used for transcription start site (TSS) analysis on the human genome (GRCh38/hg38).

## Prerequisites

- `wget`, `gunzip` -- file downloading and decompression
- `samtools` -- FASTA indexing
- GNU `parallel` -- parallel downloads
- `bigWigMerge`, `bedGraphToBigWig` ([UCSC Kent tools](https://hgdownload.soe.ucsc.edu/admin/exe/)) -- BigWig merging
- Python 3.10+ with `pyyaml` (`pip install pyyaml`)

## Sample installation with conda

```bash
# Create and activate conda environment
conda create -n procap-atlas python=3.12 -y
conda activate procap-atlas

# Install command-line tools
conda install -c bioconda -c conda-forge samtools ucsc-bigwigmerge ucsc-bedgraphtobigwig -y
# Usually not needed.
# conda install -c conda-forge parallel wget -y

# Install Python dependencies
pip install pyyaml
```

## Usage

### 1. Download data

Download scripts use relative paths and must be run from `src/download/`:

```bash
cd src/download

# Download and index GRCh38 reference genome
bash download_genome.sh

# Download plus/minus strand BigWig files (4 parallel jobs)
bash download_bigwigs.sh

# Download bidirectional peak BED files (4 parallel jobs)
bash download_peaks.sh
```

### 2. Generate experiment config

Cross-references the experiment report TSV with manifest URL files to produce a YAML config. Archived file IDs listed in `data_manifests/archive_blacklist.txt` are excluded. When bidirectional peaks are missing for an experiment, divergent peaks are used as a fallback (with a warning).

```bash
python src/preprocess/generate_config.py
```

Output: `configs/experiment_config.yaml`

### 3. Merge replicate BigWig files

Merges replicate BigWig files per experiment, keeping plus and minus strands separate. Single-replicate experiments are moved/renamed without reprocessing.

```bash
python src/preprocess/merge_bigwigs.py
```

Output: `data/processed/bigwigs/{experiment}_{biosample}_{strand}.bigWig`

### 4. Rename and organize peak files

Moves peak BED files into a processed directory with descriptive filenames including experiment ID, biosample, and peak type.

```bash
python src/preprocess/rename_peaks.py
```

Output: `data/processed/peaks/{experiment}_{biosample}_{peak_type}.bed.gz`

## Project structure

```
data_manifests/          # ENCODE file URLs, experiment metadata, archive blacklist
configs/                 # Generated YAML experiment config
src/
  download/              # Bash download scripts (run from this directory)
  preprocess/            # Python preprocessing scripts (run from repo root)
    generate_config.py   # Manifest parsing and config generation
    merge_bigwigs.py     # Replicate BigWig merging
    rename_peaks.py      # Peak file renaming
  bpnet/                 # BPNet deep learning model (planned)
  alphagenome/           # AlphaGenome analysis (planned)
data/                    # gitignored, created by scripts
  hg38.fa                # Reference genome + index
  raw/                   # Downloaded files (per replicate)
    bigwigs/
    peaks/
  processed/             # Merged/renamed files (per experiment)
    bigwigs/
    peaks/
```
