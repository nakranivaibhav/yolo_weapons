#!/usr/local/bin/python

from ultralytics import YOLO
from pathlib import Path
import shutil
import os

model_path = "/workspace/yolo_train/weapon_detection/weapon_detection_yolo11s_640/weights/best.pt"
test_images_dir = Path("/workspace/yolo_train/yolo_dataset/images/test")
test_labels_dir = Path("/workspace/yolo_train/yolo_dataset/labels/test")

temp_dataset_dir = Path("/workspace/yolo_train/yolo_dataset_dangerous_only")
temp_images_dir = temp_dataset_dir / "images" / "test"
temp_labels_dir = temp_dataset_dir / "labels" / "test"
temp_val_images_dir = temp_dataset_dir / "images" / "val"
temp_val_labels_dir = temp_dataset_dir / "labels" / "val"

print(f"\n{'='*80}")
print(f"🔍 Evaluating on DANGEROUS_TEST subset only")
print(f"Model: {model_path}")
print(f"{'='*80}\n")

temp_images_dir.mkdir(parents=True, exist_ok=True)
temp_labels_dir.mkdir(parents=True, exist_ok=True)
temp_val_images_dir.mkdir(parents=True, exist_ok=True)
temp_val_labels_dir.mkdir(parents=True, exist_ok=True)

print("📂 Filtering dangerous_test images...")
dangerous_images = list(test_images_dir.glob("dangerous_test_*.jpg"))
print(f"Found {len(dangerous_images)} dangerous_test images")

for img_path in dangerous_images:
    shutil.copy(img_path, temp_images_dir / img_path.name)
    
    label_path = test_labels_dir / img_path.with_suffix('.txt').name
    if label_path.exists():
        shutil.copy(label_path, temp_labels_dir / label_path.name)

temp_yaml = temp_dataset_dir / "data.yaml"
with open(temp_yaml, 'w') as f:
    f.write(f"""path: {temp_dataset_dir}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['Gun', 'Knife']
""")

print(f"✅ Temporary dataset created with {len(dangerous_images)} images\n")

model = YOLO(model_path)

results = model.val(
    data=str(temp_yaml),
    split='test',
    batch=16,
    imgsz=640,
    device="cpu",
    plots=True,
    save_json=True,
    project='weapon_detection',
    name='eval_dangerous_test'
)

print(f"\n{'='*80}")
print(f"✅ Dangerous test subset evaluation complete")
print(f"Results saved to: weapon_detection/eval_dangerous_test")
print(f"{'='*80}\n")

print(f"\n📊 Metrics Summary (dangerous_test only):")
print(f"Images evaluated: {len(dangerous_images)}")
print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

print(f"\n🧹 Cleaning up temporary dataset...")
shutil.rmtree(temp_dataset_dir)
print("✅ Cleanup complete")

