#!/bin/bash -l
#SBATCH --job-name=metaformer
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH -C GPU_MEM:80GB|GPU_MEM:141GB
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=48:00:00
#SBATCH --output=metaformer_train_%j.out
#SBATCH --error=metaformer_train_%j.err

set -e

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

mamba activate torch
nvidia-smi -L

NGPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
NGPUS="${NGPUS##*:}"
if [[ "${NGPUS}" == *,* ]]; then
    IFS=',' read -ra GPU_IDS <<< "${NGPUS}"
    NGPUS="${#GPU_IDS[@]}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
BATCH_SIZE="${BATCH_SIZE:-$((4 * NGPUS))}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PROFILE_BATCHES="${PROFILE_BATCHES:-0}"
PROFILE_WARMUP_BATCHES="${PROFILE_WARMUP_BATCHES:-10}"
AMP_DTYPE="${AMP_DTYPE:-none}"
EXTRA_TRAIN_ARGS=()
if [[ "${NO_SYNC_BATCHNORM:-0}" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--no_sync_batchnorm)
fi
if [[ "${COMPILE_MODEL:-0}" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--compile)
fi

torchrun --nproc_per_node="${NGPUS}" -m promoterai_torch.train \
    --checkpoint_folder models/metaformer/all_tracks/ \
    --hdf5_human_folder data/promoterai \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --profile_batches "${PROFILE_BATCHES}" \
    --profile_warmup_batches "${PROFILE_WARMUP_BATCHES}" \
    --amp_dtype "${AMP_DTYPE}" \
    --wandb_project metaformer-procap \
    --wandb_entity adamyhe-stanford-university \
    --wandb_run_name run1 \
    --log_every_batches 100 \
    --auto_resume \
    "${EXTRA_TRAIN_ARGS[@]}"
