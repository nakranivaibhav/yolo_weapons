from ultralytics.models.yolo import YOLO
import pandas as pd
import torch
import gc

DATA_DIR = "/workspace/yolo_dataset_5_jan"
MODEL_DIR = "/workspace/yolo_dangerous_weapons/models/yolo/25_dec_2025_yolo11m/weights"
PT_MODEL = f"{MODEL_DIR}/best.pt"
ENGINE_MODEL = f"{MODEL_DIR}/best.engine"
DATA_YAML = f"{DATA_DIR}/data.yaml"

results_summary = {}

print(f"\n{'='*80}")
print(f"🔍 Comparing PyTorch (.pt) vs TensorRT (.engine) on test set")
print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"📦 Evaluating PyTorch Model (.pt)")
print(f"Model: {PT_MODEL}")
print(f"{'='*80}\n")

model_pt = YOLO(PT_MODEL)
results_pt = model_pt.val(
    data=DATA_YAML,
    split='test',
    batch=16,
    imgsz=640,
    device=0,
    plots=True,
    save_json=True,
    project='weapon_detection',
    name='eval_pt_model'
)

results_summary['PyTorch (.pt)'] = {
    'Precision': results_pt.results_dict['metrics/precision(B)'],
    'Recall': results_pt.results_dict['metrics/recall(B)'],
    'mAP50': results_pt.results_dict['metrics/mAP50(B)'],
    'mAP50-95': results_pt.results_dict['metrics/mAP50-95(B)'],
}

del model_pt
del results_pt
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

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

results_summary['TensorRT (.engine)'] = {
    'Precision': results_engine.results_dict['metrics/precision(B)'],
    'Recall': results_engine.results_dict['metrics/recall(B)'],
    'mAP50': results_engine.results_dict['metrics/mAP50(B)'],
    'mAP50-95': results_engine.results_dict['metrics/mAP50-95(B)'],
}

print(f"\n{'='*80}")
print(f"📊 COMPARISON RESULTS")
print(f"{'='*80}\n")

df = pd.DataFrame(results_summary).T
df['Precision'] = df['Precision'].map('{:.4f}'.format)
df['Recall'] = df['Recall'].map('{:.4f}'.format)
df['mAP50'] = df['mAP50'].map('{:.4f}'.format)
df['mAP50-95'] = df['mAP50-95'].map('{:.4f}'.format)

print(df.to_string())

print(f"\n{'='*80}")
print(f"📈 DIFFERENCE (TensorRT - PyTorch)")
print(f"{'='*80}\n")

pt = results_summary['PyTorch (.pt)']
eng = results_summary['TensorRT (.engine)']

diff = {
    'Precision': eng['Precision'] - pt['Precision'],
    'Recall': eng['Recall'] - pt['Recall'],
    'mAP50': eng['mAP50'] - pt['mAP50'],
    'mAP50-95': eng['mAP50-95'] - pt['mAP50-95'],
}

for metric, value in diff.items():
    sign = '+' if value >= 0 else ''
    print(f"{metric}: {sign}{value:.6f}")

print(f"\n{'='*80}")
print(f"✅ Comparison complete")
print(f"{'='*80}\n")

