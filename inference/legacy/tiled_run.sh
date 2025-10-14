#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO="${PROJECT_ROOT}/data/test_video.mp4"
MODEL="${PROJECT_ROOT}/models/yolo/weapon_detection_yolo11m_640/weights/best_fp32.engine"
OUTPUT_DIR="${PROJECT_ROOT}/inference_output"

echo "Project root: ${PROJECT_ROOT}"
echo "Video: ${VIDEO}"
echo "Model: ${MODEL}"
echo ""

cd "${SCRIPT_DIR}"

uv run python tiled_tensorrt_realtime.py \
  --video "$VIDEO" \
  --model "$MODEL" \
  --out "$OUTPUT_DIR" \
  --tile_size 640 \
  --overlap 128 \
  --conf 0.25 \
  --iou 0.45 \
  --camera_fps 30 \
  --max_frames 18000 \
  --downscale 0.5 \
  --batch_tiles \
  --refine_rois \
  --roi_expand 0.3 \
  --refine_conf 0.45 \
  --track \
  --track_persist 30 \
  --min_hits 3 \
  --save_vis

