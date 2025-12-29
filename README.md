# Dangerous Weapons Detection Pipeline

Real-time gun and knife detection using YOLO11 and ConvNeXTv2.

## Pre-trained Models

All trained models are located in `/workspace/yolo_dangerous_weapons/models/`:

**Latest YOLO Model** (December 25, 2025) ⭐
- Path: `models/yolo/25_dec_2025_yolo11m/weights/best.pt`
- Architecture: YOLO11-m @ 640x640
- Best performing model after data curation

**Previous YOLO Models** (organized by date)
- `models/yolo/25_dec_2025_yolo11l/` - YOLO11-l (no significant improvement over 11m)
- `models/yolo/15_dec_2025_yolo11m/` - Earlier training run
- `models/yolo/yolo11m_5_dec/` - Initial model
- `models/yolo/augmented_27_nov/` - November baseline

**ConvNeXT Classifier**
- Path: `models/convnext_trained/best_checkpoint/`
- Accuracy: >96% precision and recall
- Used for detection refinement

## Quick Start - Export Models

### Export YOLO to TensorRT

**Note**: Export requires ~20GB disk space for dependencies. Due to space constraints, you may need to use the main project environment.

**Option 1: Using main project environment (recommended for space-constrained systems)**
```bash
cd export
source ../.venv/bin/activate
python export_yolo.py \
    ../models/yolo/25_dec_2025_yolo11m/weights/best.pt \
    /workspace/exports \
    8 \
    640 \
    ../yolo_dataset_4_dec/data.yaml
```

**Option 2: Edit export script variables**
```bash
cd export
# Edit export_yolo.sh and set:
# MODEL_PATH, OUTPUT_DIR, BATCH_SIZE, IMGSZ, DATA_YAML
./export_yolo.sh
```

**Option 3: Direct Python call**
```bash
cd export
python export_yolo.py <model.pt> [output_dir] [batch_size] [imgsz] [data.yaml]
```

**Arguments:**
- `model.pt` - Path to YOLO .pt file (required)
- `output_dir` - Export directory (optional, defaults to model's directory)
- `batch_size` - TensorRT batch size (default: 8)
- `imgsz` - Input image size (default: 640)
- `data.yaml` - Data file for INT8 calibration (default: data/yolo_dataset/data.yaml)

**Output Files:**
- `best_fp32.engine` - Full precision (slowest, best accuracy)
- `best_fp16.engine` - Half precision (default, 2x faster)
- `best_int8.engine` - INT8 quantized (3x faster)
- `best.engine` - Default (copy of FP16)

### Run Inference

```bash
cd inference
# Edit run_simple.sh and set VIDEO path
./run_simple.sh
```

## Setup

```bash
pip install uv
uv sync
```

## Project Structure

### Core Folders

**train/** - Training scripts
- `train_yolo.py` - Train YOLO11 detector
- `train_convnext.py` - Train ConvNeXT classifier
- `train_convnext.sh` - Training wrapper

**export/** - Export models to TensorRT
- `export_yolo.py` - Export YOLO (accepts 5 arguments)
- `export_yolo.sh` - Export wrapper (edit variables at top)
- `requirements-export.txt` - Export-only dependencies (if using separate env)

**inference/** - Run detection on videos
- `person_weapon_simple.py` - Person detection → Weapon detection
- `run_simple.sh` - Simple inference wrapper
- `weapon_detector_subprocess.py` - Subprocess weapon detector
- `classifier_experiment/` - YOLO + ConvNeXT classification pipeline
- `legacy/` - Tiled inference for high-res videos

**evals/** - Model evaluation
- `eval_full_test.py` - Full test set evaluation
- `eval_dangerous_test.py` - Dangerous weapons subset
- `evaluate_convnext.py` - Classifier evaluation
- `evaluate_best_model.sh` - Evaluation wrapper

**models/** - Trained weights (all models here)
- `yolo/` - YOLO models by training date
- `convnext_trained/` - ConvNeXT checkpoint
- `deyo/` - DEYO person detector

**data/** - Datasets
- `yolo_dataset/` - YOLO detection dataset
- `yolo_dataset_cls_cropped/` - ConvNeXT crops
- `convnext_dataset/` - Symlink to crops

### Analysis Folders

**confident_learning/** - Data quality & label error detection
- `yolo/` - YOLO label error analysis using cleanlab
- `convnext/` - ConvNeXT dataset cleaning with 5-fold CV

**model_interp/** - Model interpretation & visualization
- `grad_cam.py` - GradCAM visualization
- `integrated_gradients.py` - Attribution analysis
- `guided_gradcam.py` - Guided GradCAM

**captum/** - Feature attribution
- `weapon_occlusion.py` - Occlusion sensitivity
- `extract_person_crops.py` - Extract crops from videos

**classification/** - Classification utilities
- `image_classifier.py` - Classify person crops
- `create_cropped_cls.py` - Create classification dataset

**DEYO/** - RT-DETR person detector (custom fork)

**docs/** - Documentation
- `PERSON_WEAPON.md` - Person + weapon pipeline
- `temporal_filtering_experiments.md` - Tracking experiments

## Training Pipeline

```bash
# 1. Download datasets
./download_datasets.sh

# 2. Train YOLO
cd train
uv run python train_yolo.py

# 3. Export to TensorRT
cd ../export
./export_yolo.sh  # Edit MODEL_PATH first

# 4. Train ConvNeXT
cd ../train
./train_convnext.sh

# 5. Evaluate
cd ../evals
uv run python eval_full_test.py
```

## Inference Options

**Person + Weapon Detection (Single Video)**
```bash
cd inference
./run_simple.sh [VIDEO_PATH]
```

Required paths in script:
- `--deyo_model`: Path to DEYO person detector (e.g., `models/deyo/deyo-x.pt`)
- `--weapon_model`: Path to YOLO weapon model (e.g., `models/yolo/25_dec_2025_yolo11m/weights/best.pt`)
- `--video`: Input video path (passed as argument or default set in script)

**Batch Process Multiple Videos (Folder Input)**
```bash
cd inference
./run_all_videos.sh
```

Required paths in script:
- `VIDEO_DIR`: Folder containing input videos (default: `/workspace/input_videos`)
- `--deyo_model`: Path to DEYO person detector
- `--weapon_model`: Path to YOLO weapon model
- Output saved to: `inference_output_vanilla/`

**With Classification Refinement**
```bash
cd inference/classifier_experiment
./run.sh  # Edit VIDEO path
```

**Tiled for 4K Videos**
```bash
cd inference/legacy
./tiled_run.sh  # Edit VIDEO path
```

## Performance

**Detection (YOLO11-m @ 640x640)**
- FP32: ~30-40ms/frame
- FP16: ~15-25ms/frame (default)
- INT8: ~10-20ms/frame

**Classification (ConvNeXT, batch=4)**
- ~5-10ms/batch
- Accuracy: >96% precision & recall

**Real-time Capability**
- 1080p @ 30 FPS: ✅ All modes
- 4K @ 30 FPS: ✅ With 0.5x downscaling

## Methodology & Data Curation

This section details the rigorous data curation process that produced the high-quality models.

### 1. Negative Mining (~300-350 images)

To reduce false positives on common objects:
- Mined negative examples from latest test videos
- Targeted: umbrellas, sticks, phones, and other non-weapon objects
- Added to training set as hard negatives

### 2. Confident Learning with ConvNeXT

**Objective:** Identify label errors and unfeasible images

**Process:**
1. Cropped every weapon detection from all images
2. Added 4 classes: gun, knife, mobile phone, humans
3. Trained ConvNeXT classifier (achieved >96% accuracy)
4. Performed 5-fold cross-validation on all images
5. Generated out-of-sample probabilities
6. Built confusion matrix using cleanlab
7. Identified label mismatches (gun/rifle/baseball bat confusions)

**Results:**
- Found multiple label errors
- Removed noisy/ambiguous images
- Cleaned dataset significantly improved model performance

### 3. Outlier Detection (SAM3 Embeddings)

**Objective:** Find anomalous images in the 17k dataset

**Process:**
1. Used SAM3 vision encoder neck
2. Mean-pooled to get image embeddings
3. Detected 870 outliers using distance metrics
4. Manually reviewed all 870 outliers
5. Identified 15 highly problematic images
6. Retrieved top 20 most similar images for each (300 total)
7. Reviewed similarity set for dataset consistency

**Results:**
- Removed ~20-25 truly problematic images
- Fixed/relabeled the remaining images
- Ensured dataset consistency

### 4. Classification Refinement Pipeline

**Setup:**
- Person crops extracted and expanded
- Sent to YOLO for weapon detection
- Detection crops sent to ConvNeXT classifier for final filtering

**Results:**
- Reduced false positives significantly
- Some true positives reduced to flickers (inference params too tight)
- Trade-off between FP reduction and detection stability

### 5. Final Pipeline: Human + YOLO (Recommended)

**Why this works best:**
1. DEYO detects persons in frame
2. Expand person ROIs slightly
3. Run YOLO weapon detection on person crops
4. Track detections across frames

**Results:**
- Good detection performance maintained
- Most false positives eliminated
- No detection flickering issues
- Stable tracking

### 6. Model Comparison: YOLO11m vs YOLO11l

**Tested:** YOLO11-l (larger model)
**Result:** No meaningful improvements over YOLO11-m
**Decision:** Kept YOLO11-m as final model (better speed/accuracy trade-off)

### 7. Video Validation

All test videos personally reviewed frame-by-frame:
- **Classifier Pipeline:** Fewer FPs but some detection flickers
- **Human + YOLO Pipeline:** Best balance of FP reduction and stable detections
- Validated across multiple camera angles and scenarios

## Dataset Format

**YOLO (Object Detection)**
```
yolo_dataset/
├── images/{train,valid,test}/
├── labels/{train,valid,test}/
└── data.yaml
```

**ConvNeXT (Classification)**
```
yolo_dataset_cls_cropped/
├── train/{gun,knife,mobile_phone,humans}/
└── valid/{gun,knife,mobile_phone,humans}/
```

## Troubleshooting

**Models not loading?**
```bash
git lfs pull
```

**Export disk space issues?**
- Use main project venv instead of separate export env
- Clean cache: `rm -rf /root/.cache/uv/archive-v0`

**GPU out of memory (training)?**
- Reduce batch size
- Use YOLO11-s instead of YOLO11-m

**False positives in inference?**
- Use Human + YOLO pipeline (`inference/run_simple.sh`)
- Increase confidence threshold
- Enable tracking with `--min_hits 5`

**Inference too slow?**
- Use INT8 engine
- Increase downscale factor (0.5 → 0.25)
- Reduce tile size

## Requirements

- Python 3.12+
- CUDA GPU (16GB+ for training)
- uv package manager
- 20GB+ disk space for export

## License

MIT License
