# Person + Weapon Detection Pipeline

A two-stage detection pipeline that:
1. Detects persons using DEYO (RT-DETR) model
2. Detects weapons (guns/knives) in person ROIs using YOLO11

## Architecture

```
Input Video
    ↓
[DEYO Person Detection] ← GPU (DEYO's custom ultralytics)
    ↓
Person ROIs (expanded)
    ↓
[YOLO Weapon Detection] ← GPU (subprocess with fresh ultralytics)
    ↓
[Optional: ByteTrack] ← Temporal filtering
    ↓
Output Video (weapons only)
```

## Key Features

- **No NMS overhead for person detection**: DEYO is end-to-end (DETR architecture)
- **Focused weapon detection**: Only checks person ROIs, not entire frame
- **Module isolation**: DEYO and YOLO11 use separate ultralytics via subprocess
- **Optional tracking**: ByteTrack for temporal filtering and ID persistence
- **GPU acceleration**: Both models run on GPU

## Requirements

### Models (Included with Git LFS)

All models are included in the repository and managed with Git LFS:

1. **DEYO-X model**: `models/deyo/deyo-x.pt`
   - 80 COCO classes, person is class 0
   - Original source: https://github.com/ouyanghaodong/DEYO/releases

2. **YOLO11 weapon model**: `models/yolo/weapon_detection_yolo11m_640/weights/best.pt`
   - Trained on knife (class 0) and gun (class 1)

### DEYO Repository (Included)

The DEYO repository with custom ultralytics is included in `DEYO/` directory.

**Pre-applied fix**: `torch.load()` in `DEYO/ultralytics/nn/tasks.py` already includes `weights_only=False` for PyTorch 2.6+ compatibility.

### Python Dependencies
```bash
cd /root/workspace/yolo_dangerous_weapons
uv sync
```

Key packages:
- `ultralytics>=8.3.0` (for YOLO11)
- `boxmot` (for ByteTrack tracking)
- `opencv-python`
- `numpy`

## Setup

### 1. Clone Repository with LFS
```bash
git clone <repository-url>
cd yolo_dangerous_weapons

# Pull LFS files (models)
git lfs pull
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Verify Setup
```bash
# Check DEYO model
ls models/deyo/deyo-x.pt

# Check weapon model  
ls models/yolo/weapon_detection_yolo11m_640/weights/best.pt

# Check DEYO repo
ls DEYO/ultralytics/
```

## Usage

### Quick Start
```bash
cd /root/workspace/yolo_dangerous_weapons/inference
./run_simple.sh /path/to/video.mp4
```

### Manual Run
```bash
cd yolo_dangerous_weapons

uv run python inference/person_weapon_simple.py \
    --video /path/to/video.mp4 \
    --deyo_model models/deyo/deyo-x.pt \
    --weapon_model models/yolo/weapon_detection_yolo11m_640/weights/best.pt \
    --person_conf 0.3 \
    --weapon_conf 0.3 \
    --roi_expand 0.15 \
    --downscale 0.5 \
    --max_frames 999999 \
    --track \
    --track_persist 30 \
    --min_hits 3
```

## Parameters

### Detection Parameters
- `--person_conf 0.3`: Confidence threshold for person detection (lower = more persons)
- `--weapon_conf 0.3`: Confidence threshold for weapon detection
- `--roi_expand 0.15`: Expand person ROI by 15% before weapon detection
- `--downscale 0.5`: Run person detection on 50% resolution (e.g., 1920x1080 instead of 3840x2160)

### Tracking Parameters (Optional)
- `--track`: Enable ByteTrack temporal filtering
- `--min_hits 3`: Require 3 consecutive detections before confirming weapon
- `--track_persist 30`: Keep track alive for 30 frames if temporarily lost

### Other Parameters
- `--max_frames 999999`: Process entire video (stops at video end)
- `--out path/to/output.mp4`: Output video path
- `--iou 0.45`: IoU threshold for NMS (not used for DEYO, only internal YOLO)

## Output

- **Video**: Saved to `inference_output/person_weapon_simple.mp4`
- **Annotations**:
  - Red boxes: Guns
  - Yellow boxes: Knives
  - Labels: `ID:X weapon_type confidence` (if tracking enabled)

## Performance Tips

### Speed Optimization
1. **Downscaling**: Use `--downscale 0.5` for 4K videos
   - Person detection runs on 1920x1080
   - Weapon detection crops from full 3840x2160 resolution

2. **Confidence thresholds**:
   - Higher `--person_conf` = fewer ROIs = faster weapon detection
   - Higher `--weapon_conf` = fewer false positives

3. **Tracking overhead**: ByteTrack adds ~1-2ms per frame

### Memory Usage
- Sequential processing: One frame at a time
- No memory accumulation
- Safe to run on full-length videos

## Known Issues

### RTX 5090 ONNX Issue
- **Issue**: CUDA PTX compilation fails with ONNX Runtime on RTX 5090
- **Workaround**: Using subprocess with PyTorch YOLO instead of ONNX
- **Status**: Known issue with onnxruntime-gpu 1.22.0/1.23.0 on new GPUs

### Module Conflicts
- **Issue**: DEYO uses old ultralytics, YOLO11 needs new ultralytics
- **Solution**: Subprocess architecture isolates the two versions
  - Main process: DEYO with `/root/workspace/DEYO` ultralytics
  - Subprocess: YOLO11 with fresh `uv` environment ultralytics

## Architecture Details

### Why Subprocess for Weapon Detection?
```python
# Main Process (DEYO)
sys.path.insert(0, str(PROJECT_ROOT / 'DEYO'))
from ultralytics import RTDETR  # DEYO's old ultralytics
person_model = RTDETR('models/deyo/deyo-x.pt')

# Subprocess (YOLO11)
# Fresh Python process with clean ultralytics from venv
weapon_model = YOLO('models/yolo/.../best.pt')  # New ultralytics with YOLO11 support
```

**Benefits**:
- No module conflicts
- Both models run on GPU
- Clean separation of dependencies

### Communication Protocol
- **Format**: JSON over stdin/stdout
- **Data**: Base64-encoded numpy arrays (person crops)
- **Flow**: Main process → Subprocess (crop) → Subprocess returns detections

### Coordinate Transformations
1. **Person detection**: Downscaled frame (e.g., 1920x1080)
2. **ROI expansion**: Scale to full-res, expand by 15%
3. **Weapon detection**: Full-res crop
4. **Result mapping**: Scale back to person detection resolution for display

## Troubleshooting

### "WEAPON_MODEL_READY" timeout
**Cause**: Subprocess failed to load YOLO model
**Fix**: Check that `best.pt` exists and is compatible with ultralytics

### "Can't get attribute 'C3k2'"
**Cause**: DEYO's ultralytics can't load YOLO11 architecture
**Fix**: Verify subprocess is loading weapon model (not main process)

### Slow performance
**Causes**:
- CPU fallback (check GPU availability)
- Too many person detections (increase `--person_conf`)
- Large ROIs (reduce `--roi_expand`)

### Memory errors
**Cause**: Large batch or ROI accumulation
**Fix**: Process is sequential, shouldn't happen. Check subprocess isn't leaking.

## Validation

### Quick Test (60 frames)
```bash
cd /root/workspace/yolo_dangerous_weapons/inference
uv run python person_weapon_simple.py \
    --video /root/workspace/input_videos/protest.mp4 \
    --max_frames 60 \
    --track
```

### Check Output
```bash
ls -lh inference_output/person_weapon_simple.mp4
# Should see video file with detections
```

## Performance Benchmarks


## Future Improvements

2. **Batch weapon detection**: Process multiple person ROIs in one batch
3. **TensorRT**: Export both models to TensorRT for maximum speed