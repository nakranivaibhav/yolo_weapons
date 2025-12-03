#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO="${1:-/workspace/dec_2_data_collecting/clip_1_fov_30.mp4}"

cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

uv run python inference/person_weapon_simple.py \
    --video "$VIDEO" \
    --deyo_model "models/deyo/deyo-x.pt" \
    --weapon_model "/workspace/weapon_detection/augmented_27_nov/weapon_detection_yolo11m_augmented/weights/best.pt" \
    --person_conf 0.3 \
    --weapon_conf 0.35 \
    --iou 0.45 \
    --roi_expand 0.30 \
    --downscale 0.5 \
    --max_frames 999999 \
    --track \
    --track_persist 30 \
    --min_hits 5