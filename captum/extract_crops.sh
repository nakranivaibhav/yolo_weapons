#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VIDEO_PATH="/workspace/yolo_dangerous_weapons/classification/outside_left_no_weapons.mp4"
OUT_DIR="${2:-}"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: $0 <video_path> [output_dir]"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video not found: $VIDEO_PATH"
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
    OUT_DIR="/workspace/yolo_dangerous_weapons/classification/crops/outside_left"
fi

mkdir -p "$OUT_DIR"

echo "Extracting 500 person crops from entire video..."
echo "Video: $VIDEO_PATH"
echo "Output: $OUT_DIR"
echo ""

uv run python "$SCRIPT_DIR/extract_person_crops.py" \
    --video "$VIDEO_PATH" \
    --out "$OUT_DIR" \
    --model "$PROJECT_ROOT/models/deyo/deyo-x.pt" \
    --conf 0.3 \
    --expand 0.20 \
    --target_crops 500 \
    --skip 5