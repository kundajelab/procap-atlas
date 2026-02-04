#!/bin/bash

OUTDIR=../../data/raw/bigwigs/
mkdir -p $OUTDIR

cat ../../data_manifests/pl_bigwigs.txt | grep ".bigWig" | parallel -j 4 "wget --quiet -P $OUTDIR {}"
cat ../../data_manifests/mn_bigwigs.txt | grep ".bigWig" | parallel -j 4 "wget --quiet -P $OUTDIR {}"