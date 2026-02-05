#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=$REPO_ROOT/data/raw/bigwigs/
mkdir -p $OUTDIR

cat $REPO_ROOT/data_manifests/pl_bigwigs.txt | grep ".bigWig" | parallel -j 8 "wget --quiet --no-check-certificate -P $OUTDIR {}"
cat $REPO_ROOT/data_manifests/mn_bigwigs.txt | grep ".bigWig" | parallel -j 8 "wget --quiet --no-check-certificate -P $OUTDIR {}"