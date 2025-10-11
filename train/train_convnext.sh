#!/bin/bash

DATA_DIR="/workspace/yolo_dataset_cls_cropped"
OUTPUT_DIR="./weapon_classification_convnextv2"
MODEL_NAME="facebook/convnextv2-tiny-22k-224"
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=5e-5
NUM_WORKERS=8

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


