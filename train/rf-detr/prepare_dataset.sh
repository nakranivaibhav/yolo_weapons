#!/bin/bash

YOLO_DATASET="/workspace/yolo_dataset_4_dec"
COCO_DATASET="/workspace/coco_dataset"

cd "$(dirname "$0")/../.."

source .venv/bin/activate

echo "Converting YOLO dataset to COCO format..."
python train/rf-detr/prepare_coco_dataset.py \
    --yolo-dataset "$YOLO_DATASET" \
    --coco-dataset "$COCO_DATASET"

echo ""
echo "Dataset preparation complete!"
echo "COCO dataset location: $COCO_DATASET"

