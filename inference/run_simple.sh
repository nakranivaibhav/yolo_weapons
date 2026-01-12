#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

source export/.venv/bin/activate

VIDEO="${1:-/workspace/input_videos/2025_12_18_data_collecting/driving_front.mp4}"
OUTPUT_NAME=$(basename "$VIDEO" .mp4)

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

python inference/person_weapon_simple.py \
    --video "$VIDEO" \
    --out "$PROJECT_ROOT/inference_output/${OUTPUT_NAME}.mp4" \
    --deyo_model "/workspace/yolo_dangerous_weapons/models/deyo/deyo-x.engine" \
    --weapon_model "/workspace/for-svam/2026_01_05_data_collecting/original/fov_30.mp4" \
    --person_conf 0.3 \
    --weapon_conf 0.35 \
    --iou 0.45 \
    --roi_expand 0.30 \
    --downscale 0.5 \
    --max_frames 999999 \
    --track \
    --track_persist 30 \
    --min_hits 5