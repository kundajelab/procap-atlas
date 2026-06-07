# Data Download

Downloads the reference genome, ENCODE PRO-cap signal tracks, peak calls, and
GENCODE annotations used by the atlas pipeline.

## Status

These scripts are active pipeline entry points. They read curated URL manifests
from `data_manifests/` and write gitignored files under `data/`.

## Prerequisites

- `wget`
- `parallel`
- `samtools`
- Network access to ENCODE/UCSC-hosted files

The command-line tools above are available from the conda tools environment in
the root `environment.yml`.

Run commands from the repository root when possible.

## Commands

```bash
# GRCh38 reference FASTA, FASTA index, and chromosome sizes
bash src/download/download_genome.sh

# Plus/minus strand BigWig files, 8 parallel jobs
bash src/download/download_bigwigs.sh

# Bidirectional and unidirectional peak BED files, 4 parallel jobs
bash src/download/download_peaks.sh

# GENCODE v49 annotation for TSS metaplots
bash src/download/download_annotations.sh

# GTEx expression-outlier variants for PromoterAI fine-tuning
bash src/download/download_promoterai_outliers.sh
```

## Outputs

```text
data/
+-- hg38.fa
+-- hg38.fa.fai
+-- hg38.chrom.sizes
+-- gencode.v49.annotation.gff3.gz
+-- annotation/
    +-- finetune_gtex.tsv
+-- raw/
    +-- bigwigs/
    +-- peaks/
```

## Notes

- Download manifests are curated in `data_manifests/`.
- The preprocessing config is generated later by
  `src/preprocess/generate_config.py`, which cross-references the manifests and
  ENCODE experiment metadata.
- Re-running download scripts is safe for normal use, but check partial files if
  a download was interrupted.
