#!/usr/bin/env python3

import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "yolo_dataset"

os.chdir(PROJECT_ROOT)

model_name = "yolo11s.pt"

print(f"\n🚀 Starting training for {model_name}\n{'='*60}\n")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}\n")

model = YOLO(model_name)
    
model_tag = model_name.replace('.pt', '')
results = model.train(
    data=str(DATA_DIR / 'data.yaml'),
    epochs=200,
    patience=100,
    batch=32,
    imgsz=640,
    optimizer='SGD',
    momentum=0.937,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    iou=0.7,
    max_det=300,
    project='weapon_detection',
    name=f'weapon_detection_{model_tag}',
    device=0,
    plots=True
)

print(f"\n✅ Training complete for {model_tag}!")
print(f"   Location: weapon_detection/weapon_detection_{model_tag}/")
print(f"   Best model: weapon_detection/weapon_detection_{model_tag}/weights/best.pt")
print(f"\n📌 Next step: Export to TensorRT using export/export_yolo.py")
print(f"   cd ../export && uv run python export_yolo.py ../weapon_detection/weapon_detection_{model_tag}/weights/best.pt")