from ultralytics import YOLO

DATA_DIR = "/workspace/yolo_dataset_5_jan"
ENGINE_MODEL = "/workspace/yolo_dangerous_weapons/models/yolo/25_dec_2025_yolo11m/weights/best.engine"
DATA_YAML = f"{DATA_DIR}/data.yaml"

print(f"\n{'='*80}")
print(f"⚡ Evaluating TensorRT Engine (.engine)")
print(f"Model: {ENGINE_MODEL}")
print(f"{'='*80}\n")

model_engine = YOLO(ENGINE_MODEL, task='detect')
results_engine = model_engine.val(
    data=DATA_YAML,
    split='test',
    batch=16,
    imgsz=640,
    device=0,
    plots=True,
    save_json=True,
    project='weapon_detection',
    name='eval_engine_model'
)

print(f"\n{'='*80}")
print(f"📊 TensorRT Engine Results")
print(f"{'='*80}\n")
print(f"Precision: {results_engine.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {results_engine.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {results_engine.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {results_engine.results_dict['metrics/mAP50-95(B)']:.4f}")

