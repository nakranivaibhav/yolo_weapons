uv run python noise_tunneling.py \
    --weights "/workspace/weapon_detection/augmented_27_nov/weapon_detection_yolo11m_augmented/weights/best.pt" \
    --source "/workspace/input_videos/protest.mp4" \
    --limit 300 \
    --out noise_tunnel_result.mp4 \
    