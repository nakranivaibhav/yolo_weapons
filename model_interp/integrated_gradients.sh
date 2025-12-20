uv run python integrated_gradients.py \
    --weights "/workspace/weapon_detection/augmented_27_nov/weights/best.pt" \
    --source "/workspace/yolo_dangerous_weapons/model_interp/deyo_crops" \
    --out "./heatmaps/" 

