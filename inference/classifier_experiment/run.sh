#!/bin/bash

# Run the DEYO + YOLO + Classifier pipeline experiment
# Usage: ./run.sh <video_path> [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

VIDEO_PATH="${1:-/workspace/dec_2_data_collecting/clip_3_fov_30.mp4}"

cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/opencv_python.libs:${LD_LIBRARY_PATH}"

shift 2>/dev/null || true
EXTRA_ARGS="$@"

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
OUTPUT_DIR="$PROJECT_ROOT/inference_output"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "CLASSIFIER EXPERIMENT PIPELINE"
echo "=============================================================="
echo "Video: $VIDEO_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Extra args: $EXTRA_ARGS"
echo ""
echo "Usage:"
echo "  $0 <video_path> [options]"
echo ""
echo "Options:"
echo "  --track                  Enable ByteTrack tracking"
echo "  --filter_human           Filter out detections classified as 'human'"
echo "  --filter_cellphone       Filter out detections classified as 'cell_phone'"  
echo "  --use_classifier_class   Use classifier class instead of YOLO class"
echo "  --max_frames N           Process N frames (default: 60)"
echo "  --classifier_conf F      Classifier confidence threshold (default: 0.5)"
echo ""

uv run python inference/classifier_experiment/pipeline.py \
    --video "$VIDEO_PATH" \
    --deyo_model "models/deyo/deyo-x.pt" \
    --weapon_model "models/yolo/weapon_detection_yolo11m_640/weights/best.pt" \
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
    --out "$OUTPUT_DIR/${VIDEO_NAME}_classifier.mp4" \
    $EXTRA_ARGS

echo ""
echo "Done!"

