#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CROPS_DIR="$1"
OUT_DIR="${2:-}"
 
if [ -z "$CROPS_DIR" ]; then
    echo "Usage: $0 <crops_dir> [output_dir]"
    echo ""
    echo "Options (set via env vars):"
    echo "  WINDOW=32   Occlusion window size"
    echo "  STRIDE=16   Stride between windows"
    echo "  BATCH=32    Perturbations per batch"
    exit 1
fi

if [ ! -d "$CROPS_DIR" ]; then
    echo "Error: Crops directory not found: $CROPS_DIR"
    exit 1
fi

if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$(dirname "$CROPS_DIR")/occlusion"
fi

WINDOW="${WINDOW:-32}"
STRIDE="${STRIDE:-16}"
BATCH="${BATCH:-32}"

mkdir -p "$OUT_DIR"

echo "Running Occlusion attribution..."
echo "Crops: $CROPS_DIR"
echo "Output: $OUT_DIR"
echo "Window: ${WINDOW}x${WINDOW}, Stride: $STRIDE"
echo ""

uv run python "$SCRIPT_DIR/weapon_occlusion.py" \
    --crops "$CROPS_DIR" \
    --out "$OUT_DIR" \
    --model "/workspace/weapon_detection/yolo11m_5_dec/weights/best.pt" \
    --conf 0.30 \
    --size 640 \
    --window "$WINDOW" \
    --stride "$STRIDE" \
    --batch "$BATCH"
