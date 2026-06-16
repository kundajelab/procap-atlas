#!/bin/bash -l
#SBATCH --job-name=ENCSR261KBX_ccre
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --partition=akundaje
#SBATCH --mail-type=all
#SBATCH --mail-user=ayhe@stanford.edu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --array=0-1

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

conda activate "${PROCAP_ATLAS_ENV:-procap-atlas}"
export NUMBA_NUM_THREADS=12
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

biosample=ENCSR261KBX
background=ccre
# List of count, profile
declare -a heads=("count" "profile")
head="${heads[$SLURM_ARRAY_TASK_ID]}"

if [ -f "${biosample}_${background}_${head}.modisco.h5" ]; then
    echo "File $FILE exists. Skipping modisco motifs."
else    
    time uv run --project "$REPO_ROOT" --frozen --extra bpnet modisco motifs \
        -s ${biosample}_ohe.npz \
        -a ${biosample}_${background}_${head}.npz \
        -o ${biosample}_${background}_${head}.modisco.h5 \
        -n 1000000 -l 50 -w 1000 -v
fi
time uv run --project "$REPO_ROOT" --frozen --extra bpnet modisco report \
    -i ${biosample}_${background}_${head}.modisco.h5 \
    -o ${biosample}_${background}_${head}.modisco \
    -m ../../data/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt
