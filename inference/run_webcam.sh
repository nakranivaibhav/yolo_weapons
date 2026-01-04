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
    --weapon_model /Users/vaibhavnakrani/yolo_dangerous_weapons/models/yolo/15_dec_2025_yolo11m/weights/best.pt \
    --track \
    --track_persist 60 \
    --min_hits 1
    "$@"

