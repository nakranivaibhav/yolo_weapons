#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${PROJECT_ROOT}/models/convnext_trained/best_checkpoint"
DATA_DIR="${PROJECT_ROOT}/data/convnext_dataset"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results"
BATCH_SIZE=32
NUM_WORKERS=4

echo "Project root: ${PROJECT_ROOT}"
echo "Model directory: ${MODEL_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

cd "${SCRIPT_DIR}"

uv run python evaluate_convnext.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --split test

