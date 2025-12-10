uv run python grad_cam.py \
    --weights "/workspace/weapon_detection/augmented_27_nov/weights/best.pt" \
    --source "/workspace/yolo_dangerous_weapons/saliency/" \
    --out "./gradcam_heatmaps/" \