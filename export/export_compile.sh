#!/bin/bash

MODEL_PATH="/workspace/yolo_weapons_knife/weapon_classification_convnextv2/best_checkpoint"
OUTPUT_DIR="./compiled_models_convnextv2"
BATCH_SIZE=4
INPUT_SIZE=224
MODE="reduce-overhead"

export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.12/site-packages/tensorrt_libs:$LD_LIBRARY_PATH

uv run python export_torch_compile.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_SIZE \
    --mode $MODE \
    --benchmark

