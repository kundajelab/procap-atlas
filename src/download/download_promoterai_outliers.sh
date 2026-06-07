#!/bin/bash

# Downloads the GTEx expression-outlier variants used to fine-tune PromoterAI.

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR="$REPO_ROOT/data/annotation"
mkdir -p "$OUTDIR"

wget \
    https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/annotation/finetune_gtex.tsv \
    -O "$OUTDIR/finetune_gtex.tsv"
