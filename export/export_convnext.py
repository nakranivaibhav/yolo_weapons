#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import torch
import torch_tensorrt
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import json
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compile ConvNeXT model using torch.compile with TensorRT backend")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./compiled_models",
                       help="Output directory for compiled models")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Fixed batch size for compilation")
    parser.add_argument("--input_size", type=int, default=224,
                       help="Input image size (height and width)")
    parser.add_argument("--mode", type=str, default="reduce-overhead",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode")
    parser.add_argument("--dynamic", action="store_true",
                       help="Enable dynamic shapes")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark after compilation")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations for benchmark")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    return parser.parse_args()


def load_model(model_path):
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = ConvNextV2ForImageClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False
    )
    image_processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False
    )
    
    model.eval().cuda()
    
    return model, image_processor


def compile_model(model, batch_size, input_size, mode, dynamic):
    print(f"\nCompiling model with torch_tensorrt.compile (ir='dynamo')...")
    print(f"Dynamic shapes: {dynamic}")
    
    inputs = [torch.randn(batch_size, 3, input_size, input_size).cuda()]
    
    compiled_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float16, torch.float32},
        min_block_size=1,
        debug=False
    )
    
    print("Model compiled successfully!")
    
    return compiled_model, inputs


def warmup_model(model, inputs, warmup_runs=3):
    print(f"\nRunning warmup ({warmup_runs} runs)...")
    
    for i in range(warmup_runs):
        print(f"  Warmup run {i+1}/{warmup_runs}...")
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize()
    
    print("Warmup complete!")


def benchmark_model(model, batch_size, input_size, num_warmup=10, num_iterations=100):
    print(f"\nRunning benchmark (warmup={num_warmup}, iterations={num_iterations})...")
    
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).cuda()
    
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)
    
    times = np.array(times)
    
    print(f"\nBenchmark Results:")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    print(f"  FPS:    {1000/np.mean(times):.2f}")
    
    return times


def save_compiled_model(compiled_model, output_dir, batch_size, inputs):
    pt_path = os.path.join(output_dir, f"convnext_bs{batch_size}.pt")
    ep_path = os.path.join(output_dir, f"convnext_bs{batch_size}.ep")
    ts_path = os.path.join(output_dir, f"convnext_bs{batch_size}.ts")
    
    print(f"\nSaving compiled models...")
    
    print(f"  Saving as PyTorch (.pt) to: {pt_path}")
    try:
        torch.save(compiled_model, pt_path)
        print("  ✓ PyTorch (.pt) saved (simple pickle format)")
    except Exception as e:
        print(f"  Warning: Could not save .pt format: {e}")
        pt_path = None
    
    print(f"  Saving as ExportedProgram (.ep) to: {ep_path}")
    try:
        torch_tensorrt.save(compiled_model, ep_path, output_format="exported_program", inputs=inputs)
        print("  ✓ ExportedProgram (.ep) saved")
    except Exception as e:
        print(f"  Warning: Could not save ExportedProgram format: {e}")
        ep_path = None
    
    try:
        print(f"  Saving TorchScript (.ts) to: {ts_path}")
        torch_tensorrt.save(compiled_model, ts_path, output_format="torchscript", inputs=inputs)
        print("  ✓ TorchScript (.ts) saved")
    except Exception as e:
        print(f"  Warning: Could not save TorchScript format: {e}")
        print("  This is expected for models with dict outputs")
        ts_path = None
    
    return pt_path, ep_path, ts_path


def save_metadata(output_dir, model_config, batch_size, input_size, mode, dynamic):
    metadata = {
        "model_name": model_config.name_or_path if hasattr(model_config, 'name_or_path') else "unknown",
        "batch_size": batch_size,
        "input_size": input_size,
        "num_classes": model_config.num_labels,
        "id2label": model_config.id2label,
        "label2id": model_config.label2id,
        "input_shape": [batch_size, 3, input_size, input_size],
        "backend": "torch.compile with tensorrt",
        "compile_mode": mode,
        "dynamic_shapes": dynamic
    }
    
    metadata_path = os.path.join(output_dir, "export_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT compilation")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torch-TensorRT version: {torch_tensorrt.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model, image_processor = load_model(args.model_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compiled_model, inputs = compile_model(
        model,
        args.batch_size,
        args.input_size,
        args.mode,
        args.dynamic
    )
    
    warmup_model(compiled_model, inputs)
    
    if args.benchmark:
        benchmark_times = benchmark_model(
            compiled_model,
            args.batch_size,
            args.input_size,
            args.warmup,
            args.iterations
        )
        
        benchmark_file = os.path.join(args.output_dir, "benchmark_results.json")
        with open(benchmark_file, 'w') as f:
            json.dump({
                "mean_ms": float(np.mean(benchmark_times)),
                "median_ms": float(np.median(benchmark_times)),
                "min_ms": float(np.min(benchmark_times)),
                "max_ms": float(np.max(benchmark_times)),
                "std_ms": float(np.std(benchmark_times)),
                "fps": float(1000/np.mean(benchmark_times))
            }, f, indent=2)
        print(f"Benchmark results saved to: {benchmark_file}")
    
    pt_path, ep_path, ts_path = save_compiled_model(compiled_model, args.output_dir, args.batch_size, inputs)
    
    save_metadata(
        args.output_dir,
        model.config,
        args.batch_size,
        args.input_size,
        args.mode,
        args.dynamic
    )
    
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    if pt_path:
        print(f"PyTorch (.pt): {pt_path}")
    if ep_path:
        print(f"ExportedProgram (.ep): {ep_path}")
    if ts_path:
        print(f"TorchScript (.ts): {ts_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Precision: FP16 + FP32 (mixed)")
    print(f"{'='*60}")
    print("\nUsage (Python):")
    if pt_path:
        print(f"  # Option 1: PyTorch (.pt) - Simple pickle")
        print(f"  import torch")
        print(f"  model = torch.load('{pt_path}', weights_only=False).cuda().eval()")
        print(f"  model(*inputs)")
    if ep_path:
        print(f"\n  # Option 2: ExportedProgram (.ep)")
        print(f"  import torch")
        print(f"  model = torch.export.load('{ep_path}').module()")
        print(f"  model(*inputs)")
    if ts_path:
        print(f"\n  # Option 3: TorchScript (.ts)")
        print(f"  import torch")
        print(f"  model = torch.jit.load('{ts_path}').cuda()")
        print(f"  model(*inputs)")
    print(f"{'='*60}")
    
    torch._dynamo.reset()
    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

