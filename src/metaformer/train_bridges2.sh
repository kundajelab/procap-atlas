#!/bin/bash -l
#SBATCH --job-name=metaformer
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:8
#SBATCH --time=48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=ayhe@stanford.edu
#SBATCH -A bio240062p
#SBATCH --output=../../logs/metaformer/train.out
#SBATCH --error=../../logs/metaformer/train.err

set -e

conda activate bpnet

cd /jet/home/adamyhe/storage_/adamyhe/procap-atlas/src/metaformer
mkdir -p ../../logs/promoterai/
mkdir -p ../../models/metaformer/all_tracks/
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