import sys
import subprocess
import argparse
from pathlib import Path

from ultralytics.models.yolo import YOLO

def export_to_engine(model_path, batch_size=1, imgsz=640, workspace=8):
    model_path = Path(model_path)
    output_dir = model_path.parent
    
    onnx_path = output_dir / "best.onnx"
    engine_path = output_dir / "best.engine"
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Step 1: Exporting to ONNX with batch_size={batch_size}")
    model.export(format='onnx', imgsz=imgsz, dynamic=False, batch=batch_size)
    
    print(f"Step 2: Converting ONNX to TensorRT engine using trtexec")
    trtexec_cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    
    print(f"Running: {' '.join(trtexec_cmd)}")
    result = subprocess.run(trtexec_cmd, check=True, capture_output=True, text=True)
    
    print("Export completed successfully!")
    print(f"ONNX file: {onnx_path}")
    print(f"Engine file: {engine_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export YOLO model to TensorRT engine via ONNX')
    parser.add_argument('--model', type=str, 
                        default="/workspace/yolo_dangerous_weapons/models/yolo/5_jan_2026_yolo11m/weights/best.pt",
                        help='Path to YOLO model (.pt file)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size for export')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for export')
    parser.add_argument('--workspace', type=int, default=8,
                        help='Workspace size in GB for TensorRT')
    
    args = parser.parse_args()
    
    export_to_engine(
        model_path=args.model,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        workspace=args.workspace
    )
