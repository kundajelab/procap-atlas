#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=$REPO_ROOT/data/
mkdir -p $OUTDIR

wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz -O $OUTDIR/hg38.fa.gz

gunzip $OUTDIR/hg38.fa.gz
samtools faidx $OUTDIR/hg38.fa
cut -f1-2 $OUTDIR/hg38.fa.fai > $OUTDIR/hg38.chrom.sizes