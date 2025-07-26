#!/bin/bash

source /home/uglee/miniconda3/bin/activate /home/uglee/miniconda3/envs/training

# --- Configuration ---
MODEL_LOCATION="600M"
FASTA_DIR="/nashome/uglee/EnzFormer/test_fasta/first_pointmutation_600M"
OUTPUT_DIM=3
# Model architecture parameters (should match the saved models)
NUM_BLOCKS=5
N_HEAD=16
DROPOUT_RATE=0.1

USE_GPU=true

# Path to the reference PSSM .npy file for delta PSSM calculation
# Ensure this path is correct.
ENZFORMER_ROOT="/nashome/uglee/EnzFormer" # Assuming project root
REFERENCE_PSSM_PATH="$ENZFORMER_ROOT/MccB_pssm2.npy"

# Define the directory containing the checkpoints
CHECKPOINT_DIR="/nashome/uglee/EnzFormer/results/5data/three_cgl"
# Find all .pth files in the checkpoint directory
CHECKPOINTS=($(find "$CHECKPOINT_DIR" -maxdepth 1 -name "*.pth"))

# Check if any checkpoints were found
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No checkpoint files (.pth) found in $CHECKPOINT_DIR" >&2
    exit 1
fi

# Check if all found checkpoints exist (redundant check after find, but good practice)
for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "Error: Found checkpoint file seems to not exist: $ckpt" >&2
        exit 1
    fi
done

# Check if reference PSSM file exists before constructing command
if [ ! -f "$REFERENCE_PSSM_PATH" ]; then
    echo "Error: Reference PSSM file not found at $REFERENCE_PSSM_PATH" >&2
    echo "Please ensure the path is correct in the script."
    exit 1
fi

# Use an array for command arguments
CMD_ARGS=(
    "python3" 
    "evaluate_3cgl.py" 
    "$MODEL_LOCATION" 
    "$FASTA_DIR" 
    # Pass all checkpoints
    "${CHECKPOINTS[@]}" 
    "$OUTPUT_DIM" 
    "$NUM_BLOCKS" 
    # Pass model architecture args
    --n_head "$N_HEAD" 
    --dropout_rate "$DROPOUT_RATE"
    # Pass reference PSSM path for delta PSSM calculation
    --reference_pssm_npy "$REFERENCE_PSSM_PATH"
)

# Add optional --nogpu flag
if [ "$USE_GPU" = false ]; then
    CMD_ARGS+=(--nogpu)
fi

# Print and run the command
echo "Running command:"
printf "%q " "${CMD_ARGS[@]}"
echo

"${CMD_ARGS[@]}"


echo "Evaluation finished."
