#!/bin/bash

MODEL_DIR="./weapon_classification_convnextv2/best_checkpoint"
DATA_DIR="/workspace/yolo_dataset_cls_cropped"
OUTPUT_DIR="./evaluation_results_best_model"
BATCH_SIZE=32
NUM_WORKERS=4

uv run python evaluate_convnext.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --split test

