#!/bin/bash

cd "$(dirname "$0")/.."

uv run python inference/webcam_inference.py \
    --camera 0 \
    --deyo_model models/deyo/deyo-x.pt \
    --weapon_model models/yolo/25_dec_2025_yolo11m/weights/best.pt \
    --person_conf 0.3 \
    --weapon_conf 0.25 \
    --imgsz 640 \
    --width 1280 \
    --height 720 \
    "$@"

