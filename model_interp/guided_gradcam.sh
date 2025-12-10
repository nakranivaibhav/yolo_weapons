#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Input source (image folder, single image, or video)

# Output directory
OUT="${2:-$PROJECT_ROOT/model_interp/guided_gradcam_output}"



export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

uv run python guided_gradcam.py \
    --weights "/workspace/weapon_detection/augmented_27_nov/weights/best.pt" \
    --source "/workspace/yolo_dangerous_weapons/deyo_crops/" \
    --out "$OUT" \
    --model-size 640 \
    --conf-thresh 0.2 \
    --alpha 0.5 \
    --limit 100

