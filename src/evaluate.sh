#!/bin/bash

source /home/uglee/miniconda3/bin/activate /home/uglee/miniconda3/envs/training


MODEL_LOCATION="8M"
FASTA_DIR="/nashome/uglee/EnzFormer/test_fasta"
MODEL_CHECKPOINT="/nashome/uglee/EnzFormer/nopair_best_fold2/8M_blocks4_lr000034_dropout045_wd0002_earlystopped.pth"
OUTPUT_DIM=7
NUM_BLOCKS=4
USE_GPU=true

CMD="python3 evaluate.py $MODEL_LOCATION $FASTA_DIR $MODEL_CHECKPOINT $OUTPUT_DIM $NUM_BLOCKS"

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

echo "Running command: $CMD"
$CMD
