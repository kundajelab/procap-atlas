#!/bin/bash -l
#SBATCH --job-name=metaformer-ft
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gpus=4
#SBATCH -C GPU_MEM:80GB|GPU_MEM:141GB
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=48:00:00
#SBATCH --output=metaformer_finetune_%j.out
#SBATCH --error=metaformer_finetune_%j.err

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

mamba activate "${PROCAP_ATLAS_ENV:-procap-atlas}"
nvidia-smi -L

NGPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
NGPUS="${NGPUS##*:}"
if [[ "${NGPUS}" == *,* ]]; then
    IFS=',' read -ra GPU_IDS <<< "${NGPUS}"
    NGPUS="${#GPU_IDS[@]}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-models/metaformer/all_tracks/best_model.pt}"
VAR_FILE="${VAR_FILE:-data/annotation/finetune_gtex.tsv}"
FASTA_FILE="${FASTA_FILE:-data/hg38.fa}"
INPUT_LENGTH="${INPUT_LENGTH:-20480}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-6}"
AMP_DTYPE="${AMP_DTYPE:-none}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-metaformer-procap-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-adamyhe-stanford-university}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-outlier-finetune}"
EXTRA_FINETUNE_ARGS=(--auto_resume)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    EXTRA_FINETUNE_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

if (( BATCH_SIZE % NGPUS != 0 )); then
    echo "BATCH_SIZE (${BATCH_SIZE}) must be divisible by NGPUS (${NGPUS})." >&2
    exit 2
fi

torchrun --nproc_per_node="${NGPUS}" -m promoterai_torch.finetune \
    --model_checkpoint "${MODEL_CHECKPOINT}" \
    --var_file "${VAR_FILE}" \
    --fasta_file "${FASTA_FILE}" \
    --input_length "${INPUT_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --amp_dtype "${AMP_DTYPE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    "${EXTRA_FINETUNE_ARGS[@]}"
