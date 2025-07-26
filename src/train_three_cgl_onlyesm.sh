#!/bin/bash

source /home/uglee/miniconda3/bin/activate /home/uglee/miniconda3/envs/training
# run_training.sh

# Exit immediately if a command exits with a non-zero status
# Use pipefail to exit if any command in a pipeline fails
set -eo pipefail

# -------------------- User Configuration --------------------

# Replace the following variables with your actual values

# The name of the ESM BERT model: [8M, 35M, 150M, 650M, 3B, 15B]
MODEL_NAME="600M"
SAVE_DIR="/nashome/uglee/EnzFormer/results/5data/three_cgl_onlyesm_poster"
OUTPUT_DIM=3
NUM_LINEAR_LAYERS=2
BATCH_SIZE=8
LEARNING_RATE=0.0002
NUM_EPOCHS=50
WEIGHT_DECAY=0.00003     # Weight decay value for optimizer (default is 0.00001)
THRESHOLD=0.64
OPTIMIZER="AdamW"
USE_GPU=true           # Set to false to use CPU instead of GPU

# -------------------- End of User Configuration --------------------

# Define the path to the training script relative to the script's execution directory
# Assuming this script is run from the workspace root directory
TRAIN_SCRIPT="train_three_cgl_onlyesm.py"

# Check if the script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT" >&2
    exit 1
fi

# Use an array to build the command arguments for safety and clarity
cmd_args=(
    "$TRAIN_SCRIPT"
    "$MODEL_NAME"
    "$SAVE_DIR"
    "$OUTPUT_DIM"
    "$NUM_LINEAR_LAYERS"
    "$BATCH_SIZE"
    "$LEARNING_RATE"
    "$NUM_EPOCHS"
    "$THRESHOLD"
    "$OPTIMIZER"
)

# Add optional arguments to the array
if [[ -n "$WEIGHT_DECAY" ]]; then
    cmd_args+=(--weight_decay "$WEIGHT_DECAY")
fi

if [[ "$USE_GPU" == false || "$USE_GPU" == "False" ]]; then
    cmd_args+=(--nogpu)
fi

# Print the command for debugging purposes
# Use printf for safer expansion, especially if arguments contain spaces
printf "Running command: python"
printf " %q" "${cmd_args[@]}" # %q quotes arguments appropriately for the shell
printf "\n"

# Execute the command directly using the array
# The shell handles the expansion and quoting correctly here
python3 "${cmd_args[@]}"

echo "Training script finished."
