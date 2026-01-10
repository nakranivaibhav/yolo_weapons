import os
from pathlib import Path
from ultralytics.models.yolo import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = Path("/workspace/yolo_dataset_4_dec")

os.chdir(PROJECT_ROOT)

model_name = "yolo11m.pt"

run_name = f"9_jan_2026_{model_name.replace('.pt', '')}"

print(f"\n🚀 Starting training for {model_name}\n{'='*60}\n")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}\n")

model = YOLO(model_name)

model_tag = model_name.replace('.pt', '')
results = model.train(
    data=str(DATA_DIR / 'data.yaml'),
    epochs=220,
    patience=50,
    batch=64,
    imgsz=640,
    cache='disk',
    device='0,1',
    workers=20,
    
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.5,
    degrees=10.0,
    translate=0.2,
    scale=0.4,
    shear=5.0,
    perspective=0.0003,
    
    fliplr=0.5,
    flipud=0.0,
    bgr=0.0,
    
    mosaic=0.6,
    mixup=0.15,
    copy_paste=0.1,
    copy_paste_mode='flip',
    cutmix=0.0,
    auto_augment='randaugment',
    erasing=0.4,
    
    close_mosaic=20,
    dropout=0.1,
    
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    iou=0.5,
    max_det=300,
    project='weapon_detection',
    name=run_name,
    plots=True,
    
    amp=True,
    cos_lr=True,
)

print(f"\n✅ Training complete for {model_tag}!")
print(f"   Location: weapon_detection/{run_name}/")
print(f"   Best model: weapon_detection/{run_name}/weights/best.pt")
print(f"\n📌 Next step: Export to TensorRT using export/export_yolo.py")
print(f"   cd ../export && uv run python export_yolo.py ../weapon_detection/{run_name}/weights/best.pt")
