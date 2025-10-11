#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO="${PROJECT_ROOT}/input/john_wick_end.mkv"
DETECT_MODEL="${PROJECT_ROOT}/models/yolo/weapon_detection_yolo11m_640/weights/best_fp16.engine"
CLASSIFY_MODEL="${PROJECT_ROOT}/models/convnext_compiled/convnext_bs4.pt"
OUTPUT_DIR="${PROJECT_ROOT}/inference_output"

echo "Project root: ${PROJECT_ROOT}"
echo "Video: ${VIDEO}"
echo "Detection model: ${DETECT_MODEL}"
echo "Classification model: ${CLASSIFY_MODEL}"
echo ""

cd "${SCRIPT_DIR}"

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

