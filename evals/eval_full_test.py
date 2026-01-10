from ultralytics.models.yolo import YOLO

DATA_DIR = "/workspace/yolo_dataset_4_dec"
model_path = "/workspace/yolo_dangerous_weapons/models/yolo/9_jan_2026_yolo11m/weights/best.pt"
data_yaml = DATA_DIR + "/data.yaml"

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

