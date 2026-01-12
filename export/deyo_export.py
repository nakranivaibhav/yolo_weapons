import sys
from pathlib import Path

# sys.path.insert(0, "/workspace/yolo_dangerous_weapons/DEYO")


from ultralytics.models.yolo import YOLO
#  from ultralytics.models.yolo import YOLO
model_path = "/workspace/yolo_dangerous_weapons/models/yolo/5_jan_2026_yolo11m/weights/best.pt"

# print(f"Using DEYO ultralytics from: {DEYO_ROOT}")
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# print(f"Exporting to TensorRT engine: {output_path}")
model.export(format='engine', imgsz=640, half=True, dynamic=False, batch=1, workspace=8)

print("Export completed successfully!")

