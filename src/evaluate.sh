#!/bin/bash

MODEL_LOCATION="esm2_t6_8M_UR50D"
FASTA_DIR="/nashome/uglee/ECMM/tests/test_fasta"
MODEL_CHECKPOINT="/nashome/uglee/ECMM/tests/esm2_t6_8M_UR50D_2_0.001_0_0.0_earlystopped.pth"
OUTPUT_DIM=2
NUM_BLOCKS=2
LEARNING_RATE=0.001
USE_GPU=true

CMD="python3 evaluate.py $MODEL_LOCATION $FASTA_DIR $MODEL_CHECKPOINT $OUTPUT_DIM $NUM_BLOCKS"

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

echo "Running command: $CMD"
$CMD
