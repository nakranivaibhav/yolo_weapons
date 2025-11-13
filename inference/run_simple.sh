#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO="${1:-/root/workspace/input_videos/protest.mp4}"

cd "$PROJECT_ROOT"

uv run python inference/person_weapon_simple.py \
    --video "$VIDEO" \
    --deyo_model "/root/workspace/deyo_model/deyo-x.pt" \
    --weapon_model "models/yolo/weapon_detection_yolo11m_640/weights/best.pt" \
    --person_conf 0.3 \
    --weapon_conf 0.3 \
    --iou 0.45 \
    --roi_expand 0.15 \
    --downscale 0.5 \
    --max_frames 999999 \
    --track \
    --track_persist 30 \
    --min_hits 3