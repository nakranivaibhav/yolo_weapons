#!/bin/bash

VIDEO="/workspace/john_wick_end.mkv"
DETECT_MODEL="/workspace/yolo_weapons_knife/weapon_detection/weapon_detection_yolo11m_640/weights/best_fp16.engine"
CLASSIFY_MODEL="/workspace/yolo_weapons_knife/convnext_bs4.pt"
OUTPUT_DIR="./tiled_classify_out"

uv run python tiled_classification_realtime.py \
    --video "$VIDEO" \
    --detect_model "$DETECT_MODEL" \
    --classify_model "$CLASSIFY_MODEL" \
    --out "$OUTPUT_DIR" \
    --tile_size 640 \
    --detect_batch 8 \
    --classify_batch 4 \
    --conf 0.65 \
    --classify_conf 0.97 \
    --iou 0.50 \
    --camera_fps 30 \
    --max_frames 9000 \
    --downscale 0.5 \
    --roi_expand 0.2 \
    --classify_rois \
    --track \
    --track_persist 45 \
    --min_hits 5 \
    --save_vis

