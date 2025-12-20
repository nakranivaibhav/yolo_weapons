#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VIDEO_PATH="$1"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: $0 <video_path>"
    echo ""
    echo "This script runs the full pipeline:"
    echo "  1. Extract person crops from video"
    echo "  2. Run Occlusion attribution on weapon detections"
    echo ""
    echo "Output will be saved to: captum/output/<video_name>/"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video not found: $VIDEO_PATH"
    exit 1
fi

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
OUTPUT_BASE="$SCRIPT_DIR/output/$VIDEO_NAME"
CROPS_DIR="$OUTPUT_BASE/crops"
OCCLUSION_DIR="$OUTPUT_BASE/occlusion"

echo "========================================"
echo "WEAPON DETECTION OCCLUSION PIPELINE"
echo "========================================"
echo ""
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_BASE"
echo ""

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "STEP 1: Extract Person Crops"
echo "========================================"
bash "$SCRIPT_DIR/extract_crops.sh" "$VIDEO_PATH" "$CROPS_DIR"

CROP_COUNT=$(find "$CROPS_DIR" -name "*.jpg" | wc -l)
echo ""
echo "Extracted $CROP_COUNT crops"
echo ""

echo "========================================"
echo "STEP 2: Run Occlusion Attribution"
echo "========================================"
bash "$SCRIPT_DIR/run_occlusion.sh" "$CROPS_DIR" "$OCCLUSION_DIR"

echo ""
echo "========================================"
echo "DONE"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  Crops:     $CROPS_DIR"
echo "  Occlusion: $OCCLUSION_DIR"
