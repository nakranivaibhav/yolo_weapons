import sys
import os
import subprocess
import shutil
from pathlib import Path
from ultralytics.models.yolo import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def check_trtexec():
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        common_paths = [
            "/usr/local/cuda/bin/trtexec",
            "/usr/local/TensorRT/bin/trtexec",
            "/usr/src/tensorrt/bin/trtexec",
            "/opt/tensorrt/bin/trtexec"
        ]
        for path in common_paths:
            if Path(path).exists():
                print(f"✓ Found trtexec at: {path}")
                print(f"  Adding to PATH for this session...")
                os.environ['PATH'] = f"{Path(path).parent}:{os.environ['PATH']}"
                return True
        
        print("\n❌ ERROR: trtexec not found in PATH")
        print("\nNOTE: tensorrt-cu12 Python package does NOT include trtexec binary.")
        print("      trtexec is a separate command-line tool from full TensorRT distribution.")
        print("\nInstallation options:")
        print("1. Install TensorRT from NVIDIA:")
        print("   https://developer.nvidia.com/tensorrt")
        print("\n2. If using Docker, use an NVIDIA TensorRT container:")
        print("   docker pull nvcr.io/nvidia/tensorrt:24.01-py3")
        print("\n3. If TensorRT is installed, add trtexec to PATH:")
        print("   export PATH=$PATH:/path/to/TensorRT/bin")
        print("\n4. Check common locations:")
        print("   - /usr/local/cuda/bin/trtexec")
        print("   - /usr/src/tensorrt/bin/trtexec")
        return False
    else:
        print(f"✓ Using trtexec: {trtexec_path}")
    return True

if len(sys.argv) < 2:
    print("Usage: python export_yolo.py <model.pt> [output_dir] [batch_size] [imgsz]")
    print("Example: python export_yolo.py models/yolo/weapon_detection_yolo11s_640/weights/best.pt ./exports 8 640")
    sys.exit(1)

model_path = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else None
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 640

model_path = Path(model_path)
if not model_path.exists():
    print(f"Error: Model file not found: {model_path}")
    sys.exit(1)

if output_dir:
    weights_dir = Path(output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
else:
    weights_dir = model_path.parent

onnx_path = weights_dir / "best.onnx"
engine_path = weights_dir / "best.engine"

if not check_trtexec():
    sys.exit(1)

print(f"\n{'='*80}")
print(f"🚀 Exporting TensorRT Engine via ONNX (trtexec method)")
print(f"Model: {model_path}")
print(f"Image size: {imgsz}")
print(f"Batch size: {batch} (fixed)")
print(f"Dynamic: False (fixed shape optimization)")
print(f"{'='*80}\n")

print(f"📦 Step 1/2: Exporting to ONNX with batch_size={batch}...")
model = YOLO(str(model_path))
model.export(format='onnx', imgsz=imgsz, dynamic=False, batch=batch)

onnx_temp = model_path.parent / "best.onnx"
if onnx_temp.exists() and onnx_temp != onnx_path:
    onnx_temp.rename(onnx_path)
    print(f"✅ ONNX exported: {onnx_path}")

print(f"\n📦 Step 2/2: Converting ONNX to TensorRT engine using trtexec...")
trtexec_cmd = [
    "trtexec",
    f"--onnx={onnx_path}",
    f"--saveEngine={engine_path}",
    "--fp16"
]

print(f"Running: {' '.join(trtexec_cmd)}")
try:
    result = subprocess.run(trtexec_cmd, check=True, capture_output=False)
    if result.returncode == 0:
        print(f"✅ Engine exported: {engine_path}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error: trtexec failed with exit code {e.returncode}")
    print("Check the output above for details.")
    sys.exit(1)
except FileNotFoundError:
    print(f"\n❌ Error: trtexec command not found")
    print("This should have been caught earlier. Please check your installation.")
    sys.exit(1)

print(f"\n{'='*80}")
print(f"✅ Export Complete!")
print(f"{'='*80}")
print(f"\nFiles created in {weights_dir}:")
print(f"  - best.onnx   (ONNX format)")
print(f"  - best.engine (TensorRT FP16 engine)")
print(f"\nEngine optimized with:")
print(f"  - Batch size: {batch} (fixed)")
print(f"  - Precision: FP16")
print(f"  - Image size: {imgsz}x{imgsz}")
print(f"  - Method: trtexec (aligned with client compilation)")
print()

