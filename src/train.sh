#!/bin/bash

source /home/uglee/miniconda3/bin/activate /home/uglee/miniconda3/envs/training
# run_training.sh

# Exit immediately if a command exits with a non-zero status
set -e

# -------------------- User Configuration --------------------

# Replace the following variables with your actual values

# The name of the ESM BERT model: [8M, 35M, 150M, 650M, 3B, 15B]
MODEL_NAME="8M"
SAVE_DIR="/nashome/uglee/EnzFormer/results"
OUTPUT_DIM=7
NUM_BLOCKS=4
BATCH_SIZE=32
LEARNING_RATE=0.00034
NUM_EPOCHS=50
DROPOUT_RATE=0.45    # Dropout rate to use (default is 0.1)
WEIGHT_DECAY=0.002     # Weight decay value for optimizer (default is 0.00001)
N_HEAD=4
THRESHOLD=0.8
OPTIMIZER=Adam
USE_GPU=true           # Set to false to use CPU instead of GPU

# -------------------- End of User Configuration --------------------

# Construct the command to run the training script
CMD="python train.py \"$MODEL_NAME\" \"$SAVE_DIR\" $OUTPUT_DIM $NUM_BLOCKS $BATCH_SIZE $LEARNING_RATE $NUM_EPOCHS $N_HEAD $THRESHOLD $OPTIMIZER"

# Add optional arguments if they are set
if [ ! -z "$DROPOUT_RATE" ]; then
    CMD="$CMD --dropout_rate $DROPOUT_RATE"
fi

if [ ! -z "$WEIGHT_DECAY" ]; then
    CMD="$CMD --weight_decay $WEIGHT_DECAY"
fi

if [ "$USE_GPU" = false ]; then
    CMD="$CMD --nogpu"
fi

# Print the command for debugging purposes
echo "Running command: $CMD"

# Execute the command
eval $CMD
