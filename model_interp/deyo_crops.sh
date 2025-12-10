uv run python deyo_crops.py \
    --video /workspace/2025_12_04_parking_lot_weapons_video/front_camera_60fov_weapons.mp4 \
    --output_dir deyo_crops \
    --conf 0.3 \
    --downscale 0.5 \
    --roi_expand 0.0 \
    --max_frames 3000 \
    --frame_skip 30 \
    --save_format jpg