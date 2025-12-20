uv run python deyo_crops.py \
    --video /workspace/yolo_dangerous_weapons/model_interp/videos/parking_lot_front.mp4 \
    --output_dir deyo_crops \
    --conf 0.3 \
    --downscale 0.5 \
    --roi_expand 0.0 \
    --max_frames 3000 \
    --frame_skip 30 \
    --save_format jpg