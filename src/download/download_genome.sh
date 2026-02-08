#!/bin/bash

# Downloads and generates indices for hg38

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=$REPO_ROOT/data/
mkdir -p $OUTDIR

wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz -O $OUTDIR/hg38.fa.gz

gunzip $OUTDIR/hg38.fa.gz
samtools faidx $OUTDIR/hg38.fa
cut -f1-2 $OUTDIR/hg38.fa.fai > $OUTDIR/hg38.chrom.sizes

# Download mappability track

wget https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k36.Umap.MultiTrackMappability.bw -P $OUTDIR

# Download ENCODE blacklist for hg38

wget https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz -O $OUTDIR/hg38.blacklist.bed.gz