#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VIDEO_PATH="/workspace/input_videos/25_december_videos/outside_left.mp4"
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
    OUT_DIR="$SCRIPT_DIR/output/$VIDEO_NAME/"
fi

mkdir -p "$OUT_DIR"

echo "Extracting person crops..."
echo "Video: $VIDEO_PATH"
echo "Output: $OUT_DIR"
echo ""

uv run python "$SCRIPT_DIR/extract_person_crops.py" \
    --video "$VIDEO_PATH" \
    --out "$OUT_DIR" \
    --model "$PROJECT_ROOT/models/deyo/deyo-x.pt" \
    --conf 0.3 \
    --expand 0.15 \
    --max_frames 100 \
    --skip 5
