#!/bin/bash

# Downloads GENCODE v49 annotation for hg38 (GRCh38)

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR=$REPO_ROOT/data/
mkdir -p $OUTDIR

wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gff3.gz -O $OUTDIR/gencode.v49.annotation.gff3.gz
