#!/workspace/yolo_train/.venv/bin/python

from ultralytics import YOLO

model_name = "yolo11s.pt"

print(f"\n🚀 Starting training for {model_name}\n{'='*60}\n")
model = YOLO(model_name)
    
model_tag = model_name.replace('.pt', '')
results = model.train(
    data='yolo_dataset/data.yaml',
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

print(f"\n🚀 Exporting {model_tag} to TensorRT FP16...\n")
model.export(format='engine', imgsz=640, half=True, dynamic=True, batch=4, workspace=8)

print(f"\n🚀 Exporting {model_tag} to TensorRT INT8...\n")
model.export(format='engine', imgsz=640, int8=True, dynamic=True, batch=4, workspace=8, data='yolo_dataset/data.yaml')

print(f"\n✅ Training complete for {model_tag}!")
print(f"   Location: weapon_detection/weapon_detection_{model_tag}/")
print(f"   FP16: weapon_detection/weapon_detection_{model_tag}/weights/best.engine")
print(f"   INT8: weapon_detection/weapon_detection_{model_tag}/weights/best_int8.engine")