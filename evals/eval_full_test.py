#!/usr/bin/env python3

from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "yolo_dataset"

model_path = PROJECT_ROOT / "models" / "yolo" / "weapon_detection_yolo11m_640" / "weights" / "best.pt"
data_yaml = DATA_DIR / "data.yaml"

print(f"\n{'='*80}")
print(f"🔍 Evaluating on FULL test set")
print(f"Model: {model_path}")
print(f"{'='*80}\n")

model = YOLO(str(model_path))

results = model.val(
    data=str(data_yaml),
    split='test',
    batch=16,
    imgsz=640,
    device="cpu",
    plots=True,
    save_json=True,
    project='weapon_detection',
    name='eval_full_test'
)

print(f"\n{'='*80}")
print(f"✅ Full test set evaluation complete")
print(f"Results saved to: weapon_detection/eval_full_test")
print(f"{'='*80}\n")

print(f"\n📊 Metrics Summary:")
print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")

