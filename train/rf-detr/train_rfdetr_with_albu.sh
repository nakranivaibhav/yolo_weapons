#!/bin/bash

COCO_DATASET="/workspace/coco_dataset"
OUTPUT_DIR="/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano_albu_$(date +%d_%b_%Y)"

RESOLUTION=640
EPOCHS=50
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LR=1.3e-4
LR_ENCODER=8e-5
WARMUP_EPOCHS=2
WEIGHT_DECAY=3e-4
DROPOUT=0.05
DEVICE="cuda"
EMA_DECAY=0.9998

cd "$(dirname "$0")/../.."

source .venv/bin/activate

echo "========================================"
echo "RF-DETR Training with Albumentations"
echo "========================================"
echo "Augmentations applied ON-THE-FLY:"
echo "  - MotionBlur, GaussianBlur"
echo "  - GaussNoise, ISONoise"
echo "  - ImageCompression"
echo "  - RandomBrightnessContrast, RandomGamma"
echo "  - HueSaturationValue"
echo "  - RandomShadow"
echo "  - ToGray, Sharpen, CLAHE"
echo "========================================"
echo ""

python train/rf-detr/train_rfdetr_with_albumentations.py \
    --coco-dataset "$COCO_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --grad-accum-steps $GRAD_ACCUM_STEPS \
    --lr $LR \
    --lr-encoder $LR_ENCODER \
    --warmup-epochs $WARMUP_EPOCHS \
    --weight-decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --device "$DEVICE" \
    --ema-decay $EMA_DECAY \
    --use-albumentations \
    --multi-scale \
    --expanded-scales \
    --random-resize-padding \
    --use-ema

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"

