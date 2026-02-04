#!/bin/bash

OUTDIR=../../data/raw/peaks/
mkdir -p $OUTDIR

cat ../../data_manifests/bidirectional_peaks.txt | grep ".bed.gz" | parallel -j 4 "wget --quiet -P $OUTDIR {}"