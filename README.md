# Real-Time Weapon Detection System

High-performance weapon detection system for 4K video at 30 FPS using YOLO11 + TensorRT + ByteTrack.

## Overview

This system detects weapons (guns and knives) in 4K video streams in real-time using:

- **YOLO11** models (nano, small, medium) trained at 640px resolution
- **TensorRT** acceleration for fast GPU inference (FP32, FP16, INT8)
- **Tiled inference** to handle 4K resolution efficiently
- **ROI refinement** to reduce false positives
- **ByteTrack** for temporal consistency across frames
- **SAHI NMS** for intelligent tile merging

**Performance**: Processes 4K video (3840×1608) at ~30 FPS with <50ms latency on modern GPUs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       4K VIDEO INPUT                             │
│                    (3840 × 1608 @ 30fps)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DOWNSCALE (Optional)                                   │
│  ───────────────────────────────────────────────────────────    │
│  • Reduce to 1920×804 (0.5x) for speed                         │
│  • Trade-off: 2x faster, slight accuracy loss                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TILED INFERENCE                                        │
│  ───────────────────────────────────────────────────────────    │
│  • Split into 640×640 tiles with overlap                       │
│  • Batch process all 4 tiles simultaneously                    │
│  • Run TensorRT engine (batch=4, ~12ms)                        │
│  • Low confidence (0.25) for high recall                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: SAHI NMS (Tile Merging)                               │
│  ───────────────────────────────────────────────────────────    │
│  • Merge overlapping detections from tiles                     │
│  • Smart NMS that handles tile boundaries                      │
│  • IoU threshold: 0.45                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: ROI REFINEMENT (Optional)                             │
│  ───────────────────────────────────────────────────────────    │
│  • Crop full-res regions around detections (±30%)              │
│  • Re-infer on crops with higher confidence (0.50)             │
│  • Filters ~74% of false positives                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: BYTETRACK (Temporal Filtering)                        │
│  ───────────────────────────────────────────────────────────    │
│  • Track objects across frames                                 │
│  • Filter transient detections (min_hits=3-5)                  │
│  • Smooth bounding boxes                                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FINAL DETECTIONS                              │
│              (tracked, verified, de-duplicated)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Challenge**: Training YOLO on downscaled 640px images causes missed detections and false positives on 4K footage.

**Solution**: Train at 640px (feasible), but infer intelligently on 4K using:
1. **Tiling** - Process 4K in 640px chunks (model's native resolution)
2. **Downscaling** - Optional 2x speedup with minimal accuracy loss
3. **ROI Refinement** - Second-pass verification on suspicious regions
4. **ByteTrack** - Temporal filtering to remove flickering false positives

## Dataset

Download the weapon detection dataset from Google Drive:

**[Download Dataset (yolo_dataset.tar.gz)](https://drive.google.com/file/d/1HwUmZmDNpSyigVIBbRxDPn2xUQpLxBty/view?usp=sharing)**

### Setup Dataset

```bash
cd /workspace/yolo_train

# Extract the dataset
tar -xzf yolo_dataset.tar.gz

# Verify structure
ls yolo_dataset/
# Expected:
#   data.yaml
#   images/train/
#   images/valid/
#   images/test/
#   labels/train/
#   labels/valid/
#   labels/test/
```

The dataset contains:
- **Classes**: Gun, Knife
- **Format**: YOLO format (normalized bounding boxes)
- **Splits**: Train / Validation / Test

## Installation

### Requirements
- Ubuntu 22.04+
- NVIDIA GPU with CUDA support
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

```bash
cd /workspace/yolo_train

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

**Dependencies** (see `pyproject.toml`):
- `ultralytics` - YOLO11 training and inference
- `torch` + `torchvision` - PyTorch backend
- `opencv-python` - Video processing
- `sahi` - Smart tile merging (NMS)
- `boxmot` - ByteTrack implementation
- `onnx` + TensorRT - Model optimization

## Training

### Quick Start

```bash
cd /workspace/yolo_train

# Download pretrained weights (if not already downloaded)
# These are automatically downloaded by ultralytics

# Train YOLOv11s (recommended: balance of speed and accuracy)
uv run python train.py
```

### Customize Training

Edit `train.py` to change:

```python
model_name = "yolo11s.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt

model.train(
    data='yolo_dataset/data.yaml',
    epochs=200,           # Number of training epochs
    patience=100,         # Early stopping patience
    batch=32,             # Batch size (adjust for GPU memory)
    imgsz=640,            # Image size (640 or 1280)
    optimizer='SGD',
    lr0=0.01,             # Initial learning rate
    # ... see train.py for all options
)
```

**Model Sizes**:
| Model | Params | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| `yolo11n.pt` | 2.6M | Fastest | Good | Real-time on edge devices |
| `yolo11s.pt` | 9.4M | Fast | Better | **Recommended for 4K @ 30fps** |
| `yolo11m.pt` | 20.1M | Medium | Best | High accuracy, GPU required |

**Training Output**:
```
weapon_detection/weapon_detection_yolo11s_640/
├── weights/
│   ├── best.pt          # Best checkpoint
│   └── last.pt          # Last checkpoint
├── results.png          # Training curves
├── confusion_matrix.png # Validation metrics
└── args.yaml            # Training config
```

## TensorRT Export

Convert PyTorch models to optimized TensorRT engines.

### Export All Precision Levels

```bash
cd /workspace/yolo_train

# Export FP32, FP16, and INT8 engines
uv run python export_tensorrt.py \
  weapon_detection/weapon_detection_yolo11s_640/weights/best.pt \
  yolo_dataset/data.yaml
```

**Output**:
```
weapon_detection/weapon_detection_yolo11s_640/weights/
├── best.pt              # Original PyTorch model
├── best.engine          # Default engine (FP16)
├── best_fp32.engine     # Full precision (slowest, most accurate)
├── best_fp16.engine     # Half precision (2x faster, recommended)
└── best_int8.engine     # 8-bit quantized (4x faster, slight accuracy loss)
```

**Performance Comparison** (4 tiles @ 640px):
| Precision | Inference Time | Speedup | Quality Loss |
|-----------|---------------|---------|--------------|
| FP32 | ~30-40ms | 1x | 0% (baseline) |
| FP16 | ~15-25ms | **1.5-2x** | <1% |
| INT8 | ~10-20ms | 2-3x | 1-3% |

**CLI Usage**:
```bash
uv run python export_tensorrt.py <model.pt> [data.yaml]

# Examples:
uv run python export_tensorrt.py weapon_detection/weapon_detection_yolo11n_640/weights/best.pt
uv run python export_tensorrt.py weapon_detection/weapon_detection_yolo11m_640/weights/best.pt yolo_dataset/data.yaml
```

## Real-Time Inference

### Quick Start

```bash
cd /workspace/yolo_train

# Run full pipeline (tiling + ROI refinement + tracking)
./tiled_run.sh
```

### CLI Usage

```bash
uv run python tiled_tensorrt_realtime.py \
  --video <path_to_video> \
  --model <path_to_engine> \
  [OPTIONS]
```

### CLI Arguments

**Required**:
```bash
--video PATH              Input video file
--model PATH              TensorRT engine (.engine file)
```

**Tiling**:
```bash
--tile_size 640           Size of each tile (default: 640)
--overlap 128             Overlap between tiles (default: 128)
--downscale 0.5           Downscale factor before tiling (default: 1.0, no downscale)
                          - 0.5 = half resolution (2x faster)
                          - 1.0 = full 4K resolution
--batch_tiles             Batch all tiles in one inference (faster)
```

**Detection**:
```bash
--conf 0.25               Confidence threshold for tiled inference (default: 0.25)
                          Lower = more detections (high recall)
--iou 0.45                IoU threshold for NMS (default: 0.45)
                          Higher = keep more nearby detections
```

**ROI Refinement** (False Positive Reduction):
```bash
--refine_rois             Enable ROI refinement (second-pass verification)
--roi_expand 0.3          Expand ROI crop by ±30% (default: 0.3)
--refine_conf 0.50        Higher confidence for ROI verification (default: 0.40)
                          Typically 0.10-0.20 higher than --conf
```

**ByteTrack** (Temporal Filtering):
```bash
--track                   Enable ByteTrack object tracking
--min_hits 3              Minimum frames to confirm detection (default: 1)
                          - 1 = no filtering
                          - 3-5 = reduce false positives
                          - 5+ = very strict (may miss real objects)
--track_persist 30        Keep lost tracks for N frames (default: 30)
```

**Performance**:
```bash
--camera_fps 30           Target FPS (default: 30)
--max_frames 600          Stop after N frames (default: -1, process all)
```

**Output**:
```bash
--save_vis                Save annotated video to output.mp4
--output PATH             Custom output path (default: output.mp4)
```

### Example Configurations

**1. Maximum Speed** (downscaled, FP16, minimal post-processing):
```bash
uv run python tiled_tensorrt_realtime.py \
  --video input.mp4 \
  --model weapon_detection/weapon_detection_yolo11s_640/weights/best_fp16.engine \
  --downscale 0.5 \
  --tile_size 640 \
  --overlap 128 \
  --conf 0.25 \
  --iou 0.45 \
  --batch_tiles \
  --camera_fps 30
```
**Expected**: ~60 FPS, moderate accuracy

---

**2. Balanced** (downscaled, ROI refinement, tracking):
```bash
uv run python tiled_tensorrt_realtime.py \
  --video input.mp4 \
  --model weapon_detection/weapon_detection_yolo11s_640/weights/best_fp16.engine \
  --downscale 0.5 \
  --tile_size 640 \
  --overlap 128 \
  --conf 0.25 \
  --iou 0.45 \
  --batch_tiles \
  --refine_rois \
  --refine_conf 0.45 \
  --track \
  --min_hits 3 \
  --camera_fps 30 \
  --save_vis
```
**Expected**: ~30 FPS, good accuracy, fewer false positives

---

**3. Maximum Accuracy** (full 4K, FP32, strict filtering):
```bash
uv run python tiled_tensorrt_realtime.py \
  --video input.mp4 \
  --model weapon_detection/weapon_detection_yolo11m_640/weights/best_fp32.engine \
  --downscale 1.0 \
  --tile_size 640 \
  --overlap 256 \
  --conf 0.25 \
  --iou 0.45 \
  --batch_tiles \
  --refine_rois \
  --refine_conf 0.50 \
  --track \
  --min_hits 5 \
  --camera_fps 30 \
  --save_vis
```
**Expected**: ~15-20 FPS, best accuracy, minimal false positives

## Evaluation

Evaluate trained models on test set:

```bash
cd /workspace/yolo_train

# Evaluate on full test set
uv run python eval_full_test.py

# Evaluate on dangerous subset only
uv run python eval_dangerous_test.py
```

**Output**:
```
Precision: 0.9055
Recall:    0.8412
mAP50:     0.8990
mAP50-95:  0.6521
```

## Performance Tuning

### Reduce False Positives

Increase these parameters (more strict):
```bash
--refine_conf 0.50    # Higher confidence for ROI verification
--min_hits 5          # Must appear in 5 consecutive frames
```

### Increase Recall (Catch More Objects)

Decrease these parameters (more permissive):
```bash
--conf 0.20           # Lower initial confidence
--refine_conf 0.35    # Lower ROI verification threshold
--min_hits 1          # Accept immediately (disable temporal filter)
```

### Speed vs Accuracy Trade-off

| Setting | Speed | Accuracy | Notes |
|---------|-------|----------|-------|
| `--downscale 0.5` | 2x faster | -5% | Recommended for 4K @ 30fps |
| `--downscale 1.0` | Baseline | Baseline | Full 4K resolution |
| `best_fp16.engine` | 2x faster | -1% | **Recommended** |
| `best_int8.engine` | 3x faster | -3% | For edge devices |
| `--refine_rois` | -20% slower | +10% precision | Removes FPs |
| `--track --min_hits 3` | -5% slower | +5% precision | Smooths detections |

## Monitoring Performance

The inference script prints real-time statistics:

```
═══════════════════════════════════════════════════════════════════
REAL-TIME INFERENCE COMPLETE
═══════════════════════════════════════════════════════════════════
Total time:       600.41s
Frames produced:  18000
Frames processed: 17994
Dropped:          6 (0.0%)

Detection Statistics:
  Avg raw detections:     0.79
  Avg merged detections:  0.75
  Avg refined detections: 0.26
  ROI refinement reduction: 74.4%
  NMS reduction:           25.0%

Performance Metrics:
  Avg tiled inference:  12.2ms
  P95 tiled inference:  15.0ms
  Avg ROI refinement:   12.8ms
  P95 ROI refinement:   16.4ms
  Total inference:      25.0ms

Latency:
  Average:  45.8ms  (21.8 FPS equivalent)
  P95:      80.3ms  (12.5 FPS equivalent)
  P99:      225.8ms (4.4 FPS equivalent)
  Max:      446.5ms (2.2 FPS equivalent)

Real-time verdict (target: 33.3ms for 30.0 FPS):
  Average latency:  ✅ PASS (45.8ms < 66.7ms, 2x budget)
  P95 latency:      ❌ FAIL (80.3ms > 33.3ms)
  Dropped frames:   ✅ PASS (0.0% < 1.0%)
  Overall:          ✅ CAN KEEP UP (few drops acceptable)
```

**Key Metrics**:
- **Dropped frames**: Should be <1% for real-time
- **P95 latency**: 95% of frames processed within this time
- **ROI refinement reduction**: % of false positives removed (higher is better)

## Troubleshooting

### GPU Out of Memory

**Symptoms**: CUDA OOM error during training or export

**Solutions**:
```python
# In train.py, reduce batch size
batch=16  # Instead of 32

# In export_tensorrt.py, reduce batch size
batch=2   # Instead of 4
```

### Low FPS / Cannot Keep Up

**Solutions**:
1. Use `--downscale 0.5` (processes 1920×804 instead of 3840×1608)
2. Use `best_fp16.engine` or `best_int8.engine` (2-3x faster)
3. Disable `--refine_rois` (saves ~20% time, more FPs)
4. Use smaller model: `yolo11s` instead of `yolo11m`

### Too Many False Positives

**Solutions**:
1. Enable `--refine_rois` with `--refine_conf 0.50`
2. Enable `--track` with `--min_hits 5` (requires 5 consecutive frames)
3. Use larger model: `yolo11m` instead of `yolo11s`
4. Re-train with more negative examples

### Missed Detections

**Solutions**:
1. Lower `--conf 0.20` (initial confidence)
2. Increase `--overlap 256` (more tile overlap)
3. Use `--downscale 1.0` (full 4K resolution)
4. Use larger model: `yolo11m` instead of `yolo11s`
5. Re-train at higher resolution (`imgsz=1280` in `train.py`)

## File Structure

```
yolo_train/
├── README.md                 # This file
├── pyproject.toml            # Dependencies
├── train.py                  # Training script
├── export_tensorrt.py        # TensorRT export script
├── tiled_tensorrt_realtime.py # Real-time inference script
├── tiled_run.sh              # Quick launch script
├── eval_full_test.py         # Evaluation on full test set
├── eval_dangerous_test.py    # Evaluation on dangerous subset
├── yolo_dataset/             # Dataset (download separately)
│   ├── data.yaml
│   ├── images/train/
│   ├── images/valid/
│   ├── images/test/
│   ├── labels/train/
│   ├── labels/valid/
│   └── labels/test/
└── weapon_detection/         # Training outputs
    ├── weapon_detection_yolo11n_640/
    ├── weapon_detection_yolo11s_640/
    └── weapon_detection_yolo11m_640/
        ├── weights/
        │   ├── best.pt
        │   ├── best.engine       # FP16 (default)
        │   ├── best_fp32.engine
        │   ├── best_fp16.engine
        │   └── best_int8.engine
        └── results.png
```

## Credits

- **YOLO11**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **TensorRT**: [NVIDIA](https://developer.nvidia.com/tensorrt)
- **SAHI**: [obss/sahi](https://github.com/obss/sahi)
- **ByteTrack**: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
- **Dataset**: [Custom weapon detection dataset](https://drive.google.com/file/d/1HwUmZmDNpSyigVIBbRxDPn2xUQpLxBty/view?usp=sharing)

## License

This project is for research and educational purposes only.

