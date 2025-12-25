#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO_DIR="${1:-/workspace/input_videos/25_december_videos}"

cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory not found: $VIDEO_DIR"
    echo "Usage: $0 <video_directory>"
    exit 1
fi

VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" | wc -l)
echo "=============================================================="
echo "CLASSIFIER EXPERIMENT - BATCH PROCESSING"
echo "=============================================================="
echo "Video directory: $VIDEO_DIR"
echo "Videos found: $VIDEO_COUNT"
echo "=============================================================="
echo ""

PROCESSED=0
for VIDEO in "$VIDEO_DIR"/*.mp4; do
    [ -e "$VIDEO" ] || continue
    
    BASENAME=$(basename "$VIDEO" .mp4)
    OUTPUT="${PROJECT_ROOT}/inference_output/${BASENAME}_classifier.mp4"
    
    PROCESSED=$((PROCESSED + 1))
    echo "=========================================="
    echo "[$PROCESSED/$VIDEO_COUNT] Processing: $BASENAME"
    echo "Output: $OUTPUT"
    echo "=========================================="
    
    uv run python inference/classifier_experiment/pipeline.py \
        --video "$VIDEO" \
        --out "$OUTPUT" \
        --deyo_model "models/deyo/deyo-x.pt" \
        --weapon_model "/workspace/yolo_dangerous_weapons/weapon_detection/15_dec_2025_yolo11m/weights/best.pt" \
        --classifier_model "/workspace/yolo_dataset_cls_5fold/predictions/fold_0_model" \
        --person_conf 0.3 \
        --weapon_conf 0.35 \
        --classifier_conf 0.95 \
        --roi_expand 0.30 \
        --crop_expand 0.1 \
        --downscale 0.5 \
        --max_frames 999999 \
        --track \
        --track_persist 30 \
        --min_hits 5 \
        --filter_human \
        --filter_cellphone
    
    echo ""
    echo "Completed: $BASENAME"
    echo ""
done

echo "=============================================================="
echo "All $PROCESSED videos processed!"
echo "Output directory: ${PROJECT_ROOT}/inference_output/"
echo "=============================================================="

