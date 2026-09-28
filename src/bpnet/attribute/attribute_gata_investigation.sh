#!/bin/bash -l
#SBATCH --job-name=attr_gata
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=akundaje,gpu,owners
#SBATCH --time=24:00:00
#SBATCH --output=logs/bpnet_attr/attr_gata_%A.out
#SBATCH --error=logs/bpnet_attr/attr_gata_%A.err

# Compute count-head attributions for a model variant (GATA investigation).
#
# Usage:
#   sbatch src/bpnet/attribute/attribute_gata_investigation.sh MODEL_DIR PEAK_EXPERIMENT [MODEL_PREFIX]
#
# PEAK_EXPERIMENT controls which peak set is loaded (and which OHE is reused).
# MODEL_PREFIX overrides the model filename prefix (default: PEAK_EXPERIMENT).
# Needed when model files are named after the directory, not the experiment.
#
# === Full run sequence (GATA recovery investigation) ===
#
# 0. Move models into naming scheme on Sherlock:
#      models/bpnet/ENCSR261KBX_dnase/
#      models/bpnet/ENCSR261KBXtracks_ENCSR220XSMpeaks/
#      models/bpnet/ENCSR220XSMtracks_ENCSR261KBXpeaks/
#
# 1. Verify OHE files exist (needed for modisco):
#      ls attributions/bpnet/ENCSR261KBX_ohe.npz attributions/bpnet/ENCSR220XSM_ohe.npz
#      # If ENCSR220XSM_ohe.npz missing:
#      uv run --extra sherlock --frozen python src/bpnet/attribute/save_ohe.py -e ENCSR220XSM
#
# 2. Attribution (GPU, all three can run in parallel):
#      A1=$(sbatch --parsable src/bpnet/attribute/attribute_gata_investigation.sh models/bpnet/ENCSR261KBX_dnase ENCSR261KBX)
#      A2=$(sbatch --parsable src/bpnet/attribute/attribute_gata_investigation.sh models/bpnet/ENCSR261KBXtracks_ENCSR220XSMpeaks ENCSR220XSM ENCSR261KBXtracks_ENCSR220XSMpeaks)
#      A3=$(sbatch --parsable src/bpnet/attribute/attribute_gata_investigation.sh models/bpnet/ENCSR220XSMtracks_ENCSR261KBXpeaks ENCSR261KBX ENCSR220XSMtracks_ENCSR261KBXpeaks)
#
# 3. MoDISco (CPU, chained after each attribution):
#      sbatch --dependency=afterok:$A1 src/bpnet/modisco/modisco_gata_investigation.sh ENCSR261KBX_dnase ENCSR261KBX
#      sbatch --dependency=afterok:$A2 src/bpnet/modisco/modisco_gata_investigation.sh ENCSR261KBXtracks_ENCSR220XSMpeaks ENCSR220XSM
#      sbatch --dependency=afterok:$A3 src/bpnet/modisco/modisco_gata_investigation.sh ENCSR220XSMtracks_ENCSR261KBXpeaks ENCSR261KBX
#
# 4. Peak class comparison (quick, any node, no dependencies):
#      uv run --extra sherlock --frozen python src/analysis/compare_k562_peak_classes.py

MODEL_DIR="${1}"
PEAK_EXP="${2}"
MODEL_PREFIX="${3:-}"

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
nvidia-smi -L

REPO_ROOT="$SLURM_SUBMIT_DIR"
mkdir -p "$REPO_ROOT/logs/bpnet_attr"

PREFIX_FLAG=""
if [ -n "$MODEL_PREFIX" ]; then
    PREFIX_FLAG="--model-prefix $MODEL_PREFIX"
fi

uv run --project "$REPO_ROOT" --extra sherlock --frozen \
    python "$REPO_ROOT/src/bpnet/attribute/attribute_bpnet.py" \
    -e "$PEAK_EXP" \
    --model-dir "$REPO_ROOT/$MODEL_DIR" \
    --head count \
    $PREFIX_FLAG \
    -v
