#!/bin/bash

YOLO_DATASET="/workspace/yolo_dataset_4_dec"
COCO_DATASET="/workspace/coco_dataset"
OUTPUT_DIR="/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano_$(date +%d_%b_%Y)"

RESOLUTION=640
EPOCHS=50
BATCH_SIZE=64
WEIGHT_DECAY=3e-4
DROPOUT=0.05
WARMUP_EPOCHS=2

cd "$(dirname "$0")/.."

python rf-detr/train_rfdetr.py \
    --yolo-dataset "$YOLO_DATASET" \
    --coco-dataset "$COCO_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --weight-decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --warmup-epochs $WARMUP_EPOCHS

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"

