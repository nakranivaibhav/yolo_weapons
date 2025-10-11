#!/bin/bash

cd /workspace/yolo_train

# Full pipeline: Tiling + SAHI NMS + ROI Refinement + ByteTrack
# FP Reduction Strategy:
#   1. Tiled conf=0.25 (recall-first on downscaled)
#   2. ROI refine_conf=0.50 (strict verification on full-res crops)
#   3. ByteTrack min_hits=5 (must appear in 5 frames to confirm)
uv run python tiled_tensorrt_realtime.py \
  --video /workspace/yolo_infer/john_wick_end.mkv \
  --model /workspace/yolo_train/weapon_detection/weapon_detection_yolo11m_640/weights/best_fp32.engine \
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

