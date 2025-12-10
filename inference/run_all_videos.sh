#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_DIR="/workspace/2025_12_04_parking_lot_weapons_video"

cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

for VIDEO in "$VIDEO_DIR"/*.mp4; do
    BASENAME=$(basename "$VIDEO" .mp4)
    OUTPUT="${PROJECT_ROOT}/inference_output/${BASENAME}_output.mp4"
    
    echo "=========================================="
    echo "Processing: $BASENAME"
    echo "Output: $OUTPUT"
    echo "=========================================="
    
    uv run python inference/person_weapon_simple.py \
        --video "$VIDEO" \
        --out "$OUTPUT" \
        --deyo_model "models/deyo/deyo-x.pt" \
        --weapon_model "/workspace/yolo_dangerous_weapons/weapon_detection/weapon_detection_yolo11m_augmented4/weights/best.pt" \
        --person_conf 0.3 \
        --weapon_conf 0.35 \
        --iou 0.45 \
        --roi_expand 0.30 \
        --downscale 0.5 \
        --max_frames 999999 \
        --track \
        --track_persist 30 \
        --min_hits 5
    
    echo ""
    echo "Completed: $BASENAME"
    echo ""
done

echo "=========================================="
echo "All videos processed!"
echo "=========================================="
