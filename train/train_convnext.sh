#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data/convnext_dataset"
OUTPUT_DIR="${PROJECT_ROOT}/models/convnext_trained"
MODEL_NAME="facebook/convnextv2-tiny-22k-224"
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=5e-5
NUM_WORKERS=8

echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

cd "${SCRIPT_DIR}"

uv run python train_convnext.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --log_steps 50 \
    --eval_epochs 1 \
    --save_epochs 5 \
    --seed 42


