#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=$REPO_ROOT/data/raw/peaks/
mkdir -p $OUTDIR

cat $REPO_ROOT/data_manifests/bidirectional_peaks.txt | grep ".bed.gz" | parallel -j 4 "wget --quiet --no-check-certificate -P $OUTDIR {}"
cat $REPO_ROOT/data_manifests/divergent_peaks.txt | grep ".bed.gz" | parallel -j 4 "wget --quiet --no-check-certificate -P $OUTDIR {}"
cat $REPO_ROOT/data_manifests/unidirectional_peaks.txt | grep ".bed.gz" | parallel -j 4 "wget --quiet --no-check-certificate -P $OUTDIR {}"