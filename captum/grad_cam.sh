uv run python grad_cam.py \
    --weights "/workspace/weapon_detection/augmented_27_nov/weapon_detection_yolo11m_augmented/weights/best.pt" \
    --source "/workspace/input_videos/protest.mp4" \
    --out model_interp_result.mp4 \
    --limit 300