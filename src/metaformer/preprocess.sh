promoterai-torch preprocess \
    --hdf5_folder ../../../data/promoterai --tss_file tss_hg38.tsv \
    --fasta_file hg38.fa --bigwig_files hg38_tracks.tsv \
    --chrom chr1 --input_length 32768 --output_length 16384

promoterai-torch train \
    --checkpoint_folder checkpoints/run1 \
    --hdf5_human_folder data/hdf5/human \
    --input_length 20480 --output_length 4096 \
    --num_blocks 24 --model_dim 1024 --batch_size 32