#!/bin/bash -l
#SBATCH --job-name=modisco_gata
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=akundaje,normal,owners
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/bpnet_modisco/modisco_gata_%A.out
#SBATCH --error=logs/bpnet_modisco/modisco_gata_%A.err
#SBATCH -C NO_GPU

# Run modisco motifs + report on a model variant's attributions (GATA investigation).
#
# Usage:
#   sbatch src/bpnet/modisco/modisco_gata_investigation.sh MODEL_NAME PEAK_EXPERIMENT
#
# MODEL_NAME is the basename of the model dir (used as the attribution file prefix).
# PEAK_EXPERIMENT determines which OHE file to use.
#
# The three variant models:
#   sbatch modisco_gata_investigation.sh ENCSR261KBX_dnase                     ENCSR261KBX
#   sbatch modisco_gata_investigation.sh ENCSR261KBXtracks_ENCSR220XSMpeaks   ENCSR220XSM
#   sbatch modisco_gata_investigation.sh ENCSR220XSMtracks_ENCSR261KBXpeaks   ENCSR261KBX
#
# Run after the corresponding attribution job completes, e.g.:
#   ATTR_JOB=$(sbatch --parsable attribute_gata_investigation.sh ...)
#   sbatch --dependency=afterok:$ATTR_JOB modisco_gata_investigation.sh ...

MODEL_NAME="${1}"
PEAK_EXP="${2}"

ml openblas/0.3.28
ml xsimd/8.1.0
ml xz/5.8.1
ml hdf5/1.14.4
ml arrow/22.0.0
ml load py-pyarrow/18.1.0_py312
ml lz4/1.8.0
ml biology
ml htslib
ml ucsc-utils
ml gcc/14.2.0

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"
export NUMBA_NUM_THREADS=32

REPO_ROOT="$SLURM_SUBMIT_DIR"
ATTR_DIR="$REPO_ROOT/attributions/bpnet"
OUT_DIR="$REPO_ROOT/modisco/bpnet"
JASPAR="$REPO_ROOT/data/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"

mkdir -p "$OUT_DIR"
mkdir -p "$REPO_ROOT/logs/bpnet_modisco"
cd "$OUT_DIR"

time uv run --project "$REPO_ROOT" --extra sherlock --frozen modisco motifs \
    -s "$ATTR_DIR/${PEAK_EXP}_ohe.npz" \
    -a "$ATTR_DIR/${MODEL_NAME}_count.npz" \
    -o "$OUT_DIR/${MODEL_NAME}_count.modisco.h5" \
    -n 1000000 -l 50 -w 1000 -v

time uv run --project "$REPO_ROOT" --extra sherlock --frozen modisco report \
    -i "$OUT_DIR/${MODEL_NAME}_count.modisco.h5" \
    -o "$OUT_DIR/${MODEL_NAME}_count.modisco" \
    -m "$JASPAR"
