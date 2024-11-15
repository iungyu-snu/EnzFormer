#!/bin/bash

source /home/uglee/miniconda3/bin/activate /home/uglee/miniconda3/envs/training


MODEL_LOCATION="8M"
FASTA_DIR="/nashome/uglee/EnzFormer/test_fasta"
MODEL_CHECKPOINT="/nashome/uglee/EnzFormer/results/for_test/8M_EnzFormer_blocks5_lr0.0001_dropout0.3_wd_05.pth"
OUTPUT_DIM=18
NUM_BLOCKS=5
USE_GPU=true

CMD="python3 evaluate_gradient.py $MODEL_LOCATION $FASTA_DIR $MODEL_CHECKPOINT $OUTPUT_DIM $NUM_BLOCKS"

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

echo "Running command: $CMD"
$CMD
