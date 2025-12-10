import os
from pathlib import Path
import albumentations as A
from ultralytics.models.yolo import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = Path("/workspace/yolo_dataset_4_dec")

os.chdir(PROJECT_ROOT)

model_name = "yolo11m.pt"

print(f"\n🚀 Starting training for {model_name}\n{'='*60}\n")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}\n")

model = YOLO(model_name)

custom_transforms = [
    A.OneOf(
        [
            A.MotionBlur(blur_limit=(7, 25), p=1.0),
            A.Defocus(radius=(3, 7), p=1.0),
        ],
        p=0.4,
    ),
    A.OneOf(
        [
            A.GaussNoise(std_range=(0.03, 0.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ],
        p=0.3,
    ),
    A.ImageCompression(quality_range=(40, 90), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    A.Downscale(scale_range=(0.4, 0.85), p=0.2),
    A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0.5, 1, 1), p=0.2),
]

model_tag = model_name.replace('.pt', '')
results = model.train(
    data=str(DATA_DIR / 'data.yaml'),
    epochs=200,
    patience=100,
    batch=32,
    imgsz=640,
    
    augmentations=custom_transforms,
    
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    
    degrees=15.0,
    translate=0.1,      # Reduced from 0.2
    scale=0.4,
    shear=3.0,
    perspective=0.0002,
    
    fliplr=0.5,
    flipud=0.0,
    
    mosaic=0.3,         # Reduced from 0.5
    mixup=0.1,
    copy_paste=0.0,
    
    close_mosaic=15,    # Reduced from 20
    label_smoothing=0.1,
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
    name=f'weapon_detection_{model_tag}_augmented',
    device=0,
    plots=True
)

print(f"\n✅ Training complete for {model_tag}!")
print(f"   Location: weapon_detection/weapon_detection_{model_tag}_augmented/")
print(f"   Best model: weapon_detection/weapon_detection_{model_tag}_augmented/weights/best.pt")
print(f"\n📌 Next step: Export to TensorRT using export/export_yolo.py")
print(f"   cd ../export && uv run python export_yolo.py ../weapon_detection/weapon_detection_{model_tag}_augmented/weights/best.pt")