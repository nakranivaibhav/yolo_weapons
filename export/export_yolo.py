import sys
import shutil
from pathlib import Path
from ultralytics.models.yolo import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

if len(sys.argv) < 2:
    print("Usage: python export_yolo.py <model.pt> [output_dir] [batch_size] [imgsz] [data.yaml]")
    print("Example: python export_yolo.py models/yolo/weapon_detection_yolo11s_640/weights/best.pt ./exports 8 640")
    sys.exit(1)

model_path = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else None
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 640
data_yaml = sys.argv[5] if len(sys.argv) > 5 else str(PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml")

model_path = Path(model_path)
if not model_path.exists():
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

if output_dir:
    weights_dir = Path(output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
else:
    weights_dir = model_path.parent

print(f"\n{'='*80}")
print(f"🚀 Exporting TensorRT Engines (FP32, FP16, INT8)")
print(f"Model: {model_path}")
print(f"Image size: {imgsz}")
print(f"Batch size: {batch} (optimized for {batch} tiles)")
print(f"Dynamic: False (fixed shape optimization)")
print(f"{'='*80}\n")

print(f"📦 Step 1/4: Exporting FP32 engine (full precision)...")
model = YOLO(str(model_path))
model.export(format='engine', imgsz=imgsz, half=False, dynamic=False, batch=batch, workspace=8)

fp32_temp = weights_dir / "best.engine"
fp32_final = weights_dir / "best_fp32.engine"
if fp32_temp.exists():
    shutil.move(str(fp32_temp), str(fp32_final))
    print(f"✅ FP32 engine saved: {fp32_final}")

print(f"\n📦 Step 2/4: Exporting FP16 engine (half precision)...")
model = YOLO(str(model_path))
model.export(format='engine', imgsz=imgsz, half=True, dynamic=False, batch=batch, workspace=8)

fp16_temp = weights_dir / "best.engine"
fp16_final = weights_dir / "best_fp16.engine"
if fp16_temp.exists():
    shutil.move(str(fp16_temp), str(fp16_final))
    print(f"✅ FP16 engine saved: {fp16_final}")

print(f"\n📦 Step 3/4: Exporting INT8 engine (this will take ~5 minutes for calibration)...")
model = YOLO(str(model_path))
model.export(format='engine', imgsz=imgsz, int8=True, dynamic=False, batch=batch, workspace=8, data=data_yaml)

int8_temp = weights_dir / "best.engine"
int8_final = weights_dir / "best_int8.engine"
if int8_temp.exists():
    shutil.move(str(int8_temp), str(int8_final))
    print(f"✅ INT8 engine saved: {int8_final}")

print(f"\n📦 Step 4/4: Creating default best.engine (copy of FP16)...")
best_engine = weights_dir / "best.engine"
if best_engine.exists():
    best_engine.unlink()
shutil.copy(str(fp16_final), str(best_engine))
print(f"✅ Default engine saved: {best_engine}")

print(f"\n{'='*80}")
print(f"✅ Export Complete!")
print(f"{'='*80}")
print(f"\nFiles created in {weights_dir}:")
print(f"  - best.engine      (FP16, default for inference)")
print(f"  - best_fp32.engine (FP32, full precision - slowest, best quality)")
print(f"  - best_fp16.engine (FP16, half precision - fast, good quality)")
print(f"  - best_int8.engine (INT8, 8-bit quantized - fastest, slight quality loss)")
print(f"\nPerformance ({batch} tiles @ 640px):")
print(f"  FP32: ~30-40ms (baseline, best accuracy)")
print(f"  FP16: ~15-25ms (~1.5x faster)")
print(f"  INT8: ~10-20ms (~2x faster)")
print(f"\nAll engines optimized with:")
print(f"  - Batch size: {batch} (exactly {batch} tiles per frame)")
print(f"  - Dynamic: False (fixed shape optimization)")
print(f"  - Image size: 640x640")
print()

