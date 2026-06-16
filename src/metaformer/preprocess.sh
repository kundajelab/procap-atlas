#!/bin/bash -l
#SBATCH --job-name=promoterai
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH -C NO_GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=normal,akundaje,owners
#SBATCH --time=2:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/promoterai/preprocess_%a.out
#SBATCH --error=logs/promoterai/preprocess_%a.err

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

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"

mkdir -p data/promoterai
chrom=${SLURM_ARRAY_TASK_ID}
cd ../../../
promoterai-torch preprocess \
    --hdf5_folder data/promoterai \
    --tss_file data/tss_hg38.tsv \
    --fasta_file data/hg38.fa \
    --bigwig_files configs/promoterai_procap_bigwigs.tsv \
    --chrom chr${chrom} \
    --input_length 32768 \
    --output_length 16384 \
    --chunk_size 256
