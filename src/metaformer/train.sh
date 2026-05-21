#!/bin/bash -l
#SBATCH --job-name=metaformer
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH -C GPU_MEM:80GB|GPU_MEM:141GB
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=48:00:00
#SBATCH --output=../../logs/promoterai/train.out
#SBATCH --error=../../logs/promoterai/train.err

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

mkdir -p ../../models/metaformer/checkpoints/all_tracks/
torchrun --nproc_per_node=8 -m promoterai_torch.train \
    --checkpoint_folder ../../models/metaformer/all_tracks/ \
    --hdf5_human_folder ../../data/promoterai \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32 \
    --wandb_project metaformer-procap \
    --wandb_entity adamyhe-stanford-university \
    --wandb_run_name run1 \
    --log_every_batches 100 \
    --auto_resume