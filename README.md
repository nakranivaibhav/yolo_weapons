# Dangerous Weapons Detection Pipeline

A complete pipeline for training, evaluating, and deploying real-time dangerous weapons (guns and knives) detection using YOLO11 for object detection and ConvNeXTv2 for classification refinement.

## Features

- **Dual-Model Pipeline**: YOLO11 for fast object detection + ConvNeXTv2 for classification refinement
- **TensorRT Optimization**: INT8/FP16/FP32 quantization for maximum inference speed
- **Tiled Inference**: Process high-resolution (4K) videos efficiently
- **Real-time Tracking**: ByteTrack integration for object tracking across frames
- **Scalable Training**: Support for distributed training with modern optimizations

## Requirements

- Python 3.12
- CUDA-capable GPU (for training and inference)
- 16GB+ GPU memory recommended for training
- `uv` package manager
- `gdown` for dataset downloading

## Pre-trained Models

This repository includes pre-trained models ready for inference:

**YOLO Detection Models** (in `models/yolo/`):
- `weapon_detection_yolo11m_640` - YOLO11-m trained on 640x640 images (48 MB)
- `weapon_detection_yolo11s_1280` - YOLO11-s trained on 1280x1280 images (50 MB)
- Each includes TensorRT engines (FP16, FP32, INT8) for optimized inference

**ConvNeXT Classification Model** (in `models/convnext_compiled/`):
- `convnext_bs4.pt` - Compiled ConvNeXTv2 model with batch size 4 (176 MB)

⚠️ **IMPORTANT**: These models are stored using **Git LFS**. You must pull them before use:

```bash
# Install Git LFS (first time only)
sudo apt-get install git-lfs
git lfs install

# Pull all model files
git lfs pull
```

After pulling, you can skip training and go directly to inference!

### Quick Inference with Pre-trained Models

If you just want to test inference without training:

```bash
# 1. Setup environment
uv sync

# 2. Pull pre-trained models (Git LFS required!)
sudo apt-get install git-lfs
git lfs install
git lfs pull

# 3. Edit the video path in run_tiled_classification.sh
# Open inference/run_tiled_classification.sh and update line 6:
# VIDEO="/path/to/your/video.mp4"

# 4. Run inference
cd inference

./run_tiled_classification.sh  # YOLO + ConvNeXT classification
```

# Only for training.
## Quick Start

### 1. Environment Setup

```bash
# Install Python 3.12 via pyenv (if not already installed)
pyenv install 3.12

# Install uv package manager
pip install uv

# Install gdown for dataset downloads
pip install gdown

# Clone and navigate to project
cd yolo_dangerous_weapons

# Sync dependencies
uv sync
```

### 2. Download Datasets

Download both YOLO detection dataset and ConvNeXT classification dataset:

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

This will create:
- `data/yolo_dataset/` - YOLO object detection dataset (from tar.gz)
- `data/yolo_dataset_cls_cropped/` - ConvNeXT classification dataset (from zip)
- `data/convnext_dataset` - Symlink to yolo_dataset_cls_cropped for convenience

### 3. Train Models

#### Train YOLO Detection Model

```bash
cd train
uv run python train_yolo.py
```

Configuration:
- Model: YOLO11-s (small variant)
- Image size: 640x640
- Epochs: 200
- Optimizer: SGD with momentum
- Saves best model as `.pt` file (export separately)

#### Train ConvNeXT Classification Model

```bash
cd train
chmod +x train_convnext.sh
./train_convnext.sh
```

Configuration:
- Model: ConvNeXTv2-tiny-22k-224
- Epochs: 5
- Batch size: 32
- Learning rate: 5e-5
- Heavy augmentation pipeline

### 4. Evaluate Models

#### Evaluate YOLO on Full Test Set

```bash
cd evals
uv run python eval_full_test.py
```

#### Evaluate YOLO on Dangerous Subset

```bash
cd evals
uv run python eval_dangerous_test.py
```

#### Evaluate ConvNeXT Classification

```bash
cd evals
chmod +x evaluate_best_model.sh
./evaluate_best_model.sh
```

### 5. Export Models

#### Export YOLO to TensorRT

**⚠️ IMPORTANT**: Before exporting, verify the `MODEL_PATH` in `export/export_yolo.sh` matches your trained model location. By default, it points to:
```bash
MODEL_PATH="${PROJECT_ROOT}/models/yolo/weapon_detection_yolo11m_640/weights/best.pt"
```

If you trained a different model (e.g., `weapon_detection_yolo11s_640`), update line 6 in `export_yolo.sh` accordingly.

Option 1: Using shell script (recommended)
```bash
cd export
./export_yolo.sh
```

Option 2: With custom model path
```bash
cd export
./export_yolo.sh ../models/yolo/weapon_detection_yolo11s_640/weights/best.pt
```

Option 3: Direct Python call
```bash
cd export
uv run python export_yolo.py ../models/yolo/weapon_detection_yolo11s_640/weights/best.pt
```

This creates three TensorRT engines:
- `best_fp32.engine` - Full precision (best accuracy)
- `best_fp16.engine` - Half precision (2x faster)
- `best_int8.engine` - 8-bit quantized (3x faster)

**Note**: TensorRT engines are GPU-specific. If you move to a different GPU architecture (e.g., from RTX 3080 to RTX 4090), you must re-export the models.

#### Export ConvNeXT with Torch Compile

Option 1: Using shell script (recommended)
```bash
cd export
./export_convnext.sh
```

Option 2: Direct Python call
```bash
cd export
uv run python export_convnext.py \
    --model_path ../models/convnext_trained/best_checkpoint \
    --output_dir ../models/convnext_compiled \
    --batch_size 4 \
    --benchmark
```

Creates optimized models:
- `.pt` - PyTorch pickle format
- `.ep` - ExportedProgram format
- `.ts` - TorchScript format (TensorRT-optimized)

### 6. Run Inference

#### YOLO Detection Only (Fastest)

```bash
cd inference
chmod +x tiled_run.sh
./tiled_run.sh
```

Features:
- Tiled inference for 4K videos
- SAHI NMS for merging overlapping detections
- Optional ROI refinement
- ByteTrack tracking

#### YOLO + ConvNeXT Classification (Best Accuracy)

```bash
cd inference
chmod +x run_tiled_classification.sh
./run_tiled_classification.sh
```

Features:
- Two-stage pipeline: Detection → Classification
- Classification refinement reduces false positives
- Full-resolution crops for classification
- Real-time tracking

## Directory Structure

```
yolo_dangerous_weapons/
├── data/                           # Datasets (created by download_datasets.sh)
│   ├── yolo_dataset/              # YOLO detection dataset (from tar.gz)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── data.yaml
│   ├── yolo_dataset_cls_cropped/  # ConvNeXT classification dataset (from zip)
│   │   ├── train/
│   │   │   ├── gun/
│   │   │   └── knife/
│   │   └── valid/
│   │       ├── gun/
│   │       └── knife/
│   └── convnext_dataset -> yolo_dataset_cls_cropped/  # Symlink
├── train/                          # Training scripts
│   ├── train_yolo.py              # YOLO training
│   ├── train_convnext.py          # ConvNeXT training
│   └── train_convnext.sh          # ConvNeXT training script
├── evals/                          # Evaluation scripts
│   ├── eval_full_test.py          # Evaluate YOLO on full test set
│   ├── eval_dangerous_test.py     # Evaluate YOLO on dangerous subset
│   ├── evaluate_convnext.py       # Evaluate ConvNeXT classifier
│   └── evaluate_best_model.sh     # ConvNeXT evaluation script
├── export/                         # Model export scripts
│   ├── export_yolo.py             # Export YOLO to TensorRT
│   ├── export_yolo.sh             # YOLO export shell script
│   ├── export_convnext.py         # Export ConvNeXT with torch.compile
│   └── export_convnext.sh         # ConvNeXT export shell script
├── inference/                      # Inference scripts
│   ├── tiled_tensorrt_realtime.py # YOLO-only inference
│   ├── tiled_classification_realtime.py  # YOLO + ConvNeXT inference
│   ├── tiled_run.sh               # YOLO inference script
│   └── run_tiled_classification.sh # Classification pipeline script
├── models/                         # Trained models (created during training)
│   ├── yolo/                      # YOLO models and weights
│   │   └── weapon_detection_*/
│   │       └── weights/
│   │           ├── best.pt        # PyTorch model
│   │           ├── best_fp16.engine  # TensorRT FP16
│   │           ├── best_fp32.engine  # TensorRT FP32
│   │           └── best_int8.engine  # TensorRT INT8
│   ├── convnext_trained/          # ConvNeXT trained checkpoints
│   └── convnext_compiled/         # ConvNeXT compiled models
│       └── convnext_bs4.pt        # Compiled model (batch size 4)
├── download_datasets.sh            # Dataset download script
├── pyproject.toml                  # Python dependencies
├── uv.lock                         # Locked dependencies
└── README.md                       # This file
```

## Pipeline Workflows

### Training Workflow

1. Download datasets → 2. Train YOLO → 3. Export YOLO to TensorRT → 4. Train ConvNeXT → 5. Export ConvNeXT → 6. Evaluate both models

### Inference Workflow

1. Ensure models are exported → 2. Run tiled inference → 3. Get detections with tracking

### Classification Pipeline

1. YOLO detects weapons on downscaled frame (fast)
2. Extract ROIs from full-resolution frame
3. ConvNeXT classifies each ROI (accurate)
4. Filter by classification confidence
5. Track objects across frames

## Performance

### YOLO Detection
- **FP32**: ~30-40ms per frame (baseline)
- **FP16**: ~15-25ms per frame (~1.5-2x faster)
- **INT8**: ~10-20ms per frame (~2-3x faster)

### ConvNeXT Classification
- **Torch Compile + TensorRT**: ~5-10ms per batch (4 crops)

### Real-time Performance
- **1080p @ 30 FPS**: ✅ Real-time capable (all modes)
- **4K @ 30 FPS**: ✅ Real-time capable (with 0.5x downscaling)

## Configuration

### Key Parameters

#### Detection Confidence
- `--conf 0.25` - Initial detection threshold (lower = higher recall)
- `--iou 0.45` - NMS IoU threshold

#### Classification Confidence
- `--classify_conf 0.97` - Classification threshold (higher = fewer false positives)

#### Tracking Parameters
- `--track` - Enable ByteTrack
- `--min_hits 5` - Minimum detections before confirming track
- `--track_persist 45` - Frames to keep track after disappearance

#### Tiling Parameters
- `--tile_size 640` - Tile size for detection
- `--downscale 0.5` - Downscale factor for input video
- `--detect_batch 8` - Detection batch size
- `--classify_batch 4` - Classification batch size

## Dataset Format

### YOLO Dataset

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── image1.txt  # YOLO format: class x_center y_center width height
│   │   └── image2.txt
│   ├── valid/
│   └── test/
└── data.yaml
```

**data.yaml** format:
```yaml
path: /path/to/yolo_dataset
train: images/train
val: images/valid
test: images/test

nc: 2
names: ['Gun', 'Knife']
```

### ConvNeXT Dataset

```
yolo_dataset_cls_cropped/  (or use symlink: convnext_dataset/)
├── train/
│   ├── gun/
│   │   ├── gun_001.jpg
│   │   └── gun_002.jpg
│   └── knife/
│       ├── knife_001.jpg
│       └── knife_002.jpg
└── valid/
    ├── gun/
    └── knife/
```

**Note**: The actual folder is `yolo_dataset_cls_cropped`, but a symlink `convnext_dataset` is created for convenience.

## Troubleshooting

### Dataset Download Issues

**Problem 1**: Download fails or archive file is corrupted

**Solutions**:
1. The script will automatically retry with alternative methods
2. If automatic download fails, manually download:
   - YOLO dataset (tar.gz): https://drive.google.com/file/d/1HwUmZmDNpSyigVIBbRxDPn2xUQpLxBty/view?usp=drive_link
   - ConvNeXT dataset (zip): https://drive.google.com/file/d/1IRommjmeYrKsy0K5qLlrUR309HZfv5OY/view?usp=sharing
3. Place downloaded files in `data/` folder:
   - YOLO: `data/yolo_dataset.tar.gz`
   - ConvNeXT: `data/yolo_dataset_cls_cropped.zip`
4. Run `./download_datasets.sh` again to extract

**Clean up corrupted downloads**:
```bash
rm -f data/yolo_dataset.tar.gz data/yolo_dataset_cls_cropped.zip
./download_datasets.sh
```

**Problem 2**: macOS metadata files (._* files) after extraction

If you see files like `._data.yaml` or warnings about "Ignoring unknown extended header keyword", don't worry - these are harmless macOS resource fork files. 

**The download script automatically removes these files during extraction**, so you shouldn't see them. If they appear, simply run:

```bash
find data -name "._*" -type f -delete
find data -name ".DS_Store" -type f -delete
```

**Note**: The tar extraction warnings about "LIBARCHIVE.xattr" are harmless and can be ignored. They're just Apple-specific metadata that Linux ignores.

### GPU Out of Memory
- Reduce batch size in training scripts
- Use smaller YOLO variant (yolo11n instead of yolo11s)
- Enable gradient checkpointing

### TensorRT Export Fails
- Ensure CUDA and TensorRT are properly installed
- Check GPU compatibility
- Try FP16 before INT8

### Inference Too Slow
- Use INT8 quantized models
- Increase downscale factor (0.5 → 0.25)
- Reduce tile size
- Disable classification refinement

### False Positives
- Increase `--classify_conf` threshold
- Enable tracking with `--min_hits 5`
- Use classification pipeline

### Git LFS Model Files

**Problem**: Models show "invalid load key" error, "FileNotFoundError", or appear as ASCII text

**Cause**: **All pre-trained models** (YOLO TensorRT engines + ConvNeXT) are stored using Git LFS. You only have pointer files, not the actual models.

**Solution - Pull All Models**:

```bash
# Install Git LFS (one-time setup)
sudo apt-get install git-lfs
git lfs install

# Pull ALL model files (~275 MB total)
git lfs pull

# Or pull specific models only:
git lfs pull --include="models/yolo/**/*.engine"  # YOLO TensorRT engines only
git lfs pull --include="models/convnext_compiled/*.pt"  # ConvNeXT only
```

**Check if LFS files are downloaded**:
```bash
# Should show binary data, not "version https://git-lfs.github.com"
file models/yolo/weapon_detection_yolo11m_640/weights/best_fp16.engine
```

**Alternative**: Train your own models from scratch (see training sections).

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{dangerous_weapons_detection,
  title={Dangerous Weapons Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yolo_dangerous_weapons}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SAHI](https://github.com/obss/sahi)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [TensorRT](https://developer.nvidia.com/tensorrt)

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and discussions
