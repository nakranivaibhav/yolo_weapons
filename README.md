# 🔫 Real-Time Dangerous Weapons Detection System

> **Production-ready weapon detection pipeline built in 3 months** — from initial prototype to deployment-ready models with 96%+ precision and recall.

<p align="center">
  <img src="https://img.shields.io/badge/Development-3%20Months-brightgreen" alt="Timeline"/>
  <img src="https://img.shields.io/badge/mAP50-95%25+-blue" alt="mAP50"/>
  <img src="https://img.shields.io/badge/FPS-30%2B-orange" alt="FPS"/>
  <img src="https://img.shields.io/badge/TensorRT-Optimized-76B900" alt="TensorRT"/>
</p>

---

## 📋 Executive Summary

Complete end-to-end weapon detection system capable of detecting **guns, knives, rifles, and baseball bats** in real-time video streams. The system uses a two-stage architecture: person detection followed by weapon detection within person ROIs, dramatically reducing false positives while maintaining high recall.

### Key Achievements

| Metric | Value |
|--------|-------|
| **Development Time** | Oct 11, 2025 → Jan 16, 2026 (~3 months) |
| **Classification Accuracy** | >96% precision & recall |
| **Detection mAP@50** | ~95% |
| **Real-time Performance** | 30+ FPS on 1080p (TensorRT) |
| **False Positive Reduction** | >90% vs baseline |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              DEYO (RT-DETR) Person Detection                     │
│              • End-to-end transformer (no NMS overhead)          │
│              • 80 COCO classes, person = class 0                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Person ROIs (expanded 15%)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              YOLO11-m Weapon Detection                           │
│              • 4 classes: knife, gun, rifle, baseball_bat        │
│              • Trained on curated 17k+ image dataset             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ByteTrack Temporal Filtering                        │
│              • 4-second memory buffer                            │
│              • 5-7 frame confirmation threshold                  │
│              • Eliminates flickering detections                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANNOTATED OUTPUT                              │
│              • Red boxes: Guns/Rifles                            │
│              • Yellow boxes: Knives                              │
│              • Track IDs with confidence scores                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd yolo_dangerous_weapons

# Pull model weights (Git LFS)
git lfs pull

# Install dependencies
pip install uv
uv sync
```

### Run Inference

```bash
cd inference
./run_simple.sh /path/to/video.mp4
```

### Export to TensorRT (Production)

```bash
cd export
python export_yolo.py \
    ../models/yolo/25_dec_2025_yolo11m/weights/best.pt \
    /workspace/exports \
    8 \
    640 \
    ../yolo_dataset_4_dec/data.yaml
```

---

## 📊 Development Timeline

A rapid iteration cycle with continuous improvements based on real-world testing.

```
Nov 13   ────────────────────────────────────────────────────────────────────────────►
         │
         ├─ New DEYO + YOLO pipeline architecture
         │
Nov 14   ├─ E2E inference scripts, initial experiments
         │
Nov 24-27├─ YouTube/GDD classification, experiment refinements
         │
Dec 3    ├─ Captum model interpretability (GradCAM, Occlusion)
         │
Dec 10   ├─ Confident Learning + DEYO integration
         │  Label error detection with 5-fold cross-validation
         │
Dec 12   ├─ Two rounds of confident learning cleanup
         │  ~300 problematic images identified and fixed
         │
Dec 20-25├─ Major data curation sprint:
         │  • SAM3 embedding outlier detection (870 outliers → 20 removed)
         │  • Hard negative mining (person crops, mobile phones, deployment false positives)
         │  • Best recall achieved on private test set
         │
Dec 29   ├─ TensorRT export optimization
         │
Jan 1    ├─ DEYO ultralytics backend fixes for TRT
         │
Jan 4    ├─ Webcam inference + temporal parameter tuning
         │
Jan 6    ├─ Balanced dataset training, minority class improvement
         │
Jan 7    ├─ Multi-GPU training setup, RF-DETR experiments
         │
Jan 10-12├─ RF-DETR training (74% mAP vs YOLO 70%)
         │  Umbrella negatives added
         │
Jan 13   ├─ Additional negative mining checkpoints
         │
Jan 16   ├─ Final evaluation plots, attention visualization
         │  Manifold studies for model understanding
         │
         ▼
       PRODUCTION READY
```

---

## 🧪 State-of-the-Art Data Curation Pipeline

### 1. Confident Learning with 5-Fold Cross-Validation

Used **cleanlab** methodology with **ConvNeXTv2** classifier to systematically identify and fix label errors.

**Process:**
1. Cropped every weapon detection from all images
2. Split crops into 5 folds
3. Train ConvNeXTv2 on 4 folds, predict on held-out fold
4. Repeat for all folds (every image gets out-of-sample prediction)
5. Build confusion matrix using cleanlab, identify systematic mismatches
6. Manual review of flagged images

**Results:**
- Identified gun/rifle/baseball bat confusions
- Removed ambiguous and mislabeled images
- Two complete rounds of cleanup performed

```python
# confident_learning/convnext/train_convnext_cv_folds.py
# Automated 5-fold CV training with prediction collection
```

### 2. ConvNeXT Classifier for Verification

Trained a **ConvNeXTv2-tiny** classifier to verify weapon detections.

**Classes:** gun, knife, mobile phone, humans

**Performance:** >96% precision and recall

**Use Cases:**
- Post-detection filtering
- Label verification during data curation
- Confidence boosting for edge cases

### 3. SAM3 Embedding Outlier Detection

Novel approach using **Segment Anything Model 3** vision encoder for anomaly detection.

**Process:**
1. Extract SAM3 embeddings (mean-pooled) for all 17k images
2. Compute distance metrics in embedding space
3. Identify 870 statistical outliers
4. Manual review of outliers and their 20 nearest neighbors

**Results:**
- Found 15 highly problematic images
- Retrieved similar images for consistency check
- Removed ~20-25 truly problematic samples

### 4. Hard Negative Mining

Systematically reduced false positives through multi-source negative mining.

**Sources:**
- **~2000 person crops** from public datasets (non-weapon carrying individuals)
- **400 mobile phone images** (commonly confused with weapons)
- **Deployment false positives** — real-world false positives captured during actual system testing

**Impact:** Dramatically reduced false positives on umbrellas, sticks, phones, and other elongated objects

### 5. Monte Carlo Influence Functions

Advanced technique to identify which training images most impact model performance.

```python
# monte_carlo_influence/monte_carlo_influence.py
# 20 runs × random subsets → influence score per image
```

**Metrics tracked:** precision, recall, F1, mAP50, mAP50-95 at epochs 50 & 100

---

## 🔬 Model Interpretability Suite

Comprehensive tools to understand model decisions — critical for security applications.

### GradCAM Visualization

```bash
cd model_interp
./grad_cam.sh
```

Generates heatmaps showing which image regions drive detections.

### Integrated Gradients

```bash
./integrated_gradients.sh
```

Attribution method for understanding feature importance with smoothgrad noise reduction.

### Occlusion Sensitivity

```bash
cd captum
python weapon_occlusion.py --crops ./crops --out ./output
```

Sliding window occlusion to identify critical regions for each detection.

---

## 🏋️ Training Infrastructure

### YOLO11 Training

```bash
cd train/yolo
python train_yolo.py
```

**Features:**
- Multi-GPU DDP support
- Custom Albumentations augmentation pipeline
- Motion blur, defocus, ISO noise simulation
- Image compression artifacts
- Random shadows and brightness

### RF-DETR Training (Transformer Alternative)

```bash
cd train/rf-detr
./train_rfdetr.sh
```

**Comparison:**

| Model | Architecture | mAP | Small Objects | Training Time |
|-------|-------------|-----|---------------|---------------|
| YOLO11-m | CNN | ~70% | Good | 2-3 hours |
| RF-DETR Nano | Transformer | ~74% | Excellent | 8-12 hours |

### ConvNeXT Classifier Training

```bash
cd train/convnext
./train_convnext.sh
```

Uses Hugging Face Transformers with custom augmentation pipeline.

---

## 📂 Project Structure

```
.
├── inference/                    # Production inference scripts
│   ├── person_weapon_simple.py   # Main two-stage pipeline
│   ├── weapon_detector_subprocess.py  # GPU subprocess for YOLO
│   ├── webcam_inference.py       # Real-time webcam demo
│   └── run_simple.sh             # Quick start wrapper
│
├── train/                        # Training pipelines
│   ├── yolo/                     # YOLO11 training
│   ├── rf-detr/                  # RF-DETR transformer training
│   └── convnext/                 # ConvNeXTv2 classifier
│
├── confident_learning/           # Data quality tools
│   ├── yolo/                     # YOLO label error detection
│   └── convnext/                 # Classifier-based cleaning
│
├── model_interp/                 # Interpretability
│   ├── grad_cam.py               # GradCAM visualization
│   ├── integrated_gradients.py   # Attribution analysis
│   └── guided_gradcam.py         # Guided GradCAM
│
├── captum/                       # Feature attribution
│   ├── weapon_occlusion.py       # Occlusion sensitivity
│   └── extract_person_crops.py   # Crop extraction utility
│
├── monte_carlo_influence/        # Influence functions
│   └── monte_carlo_influence.py  # Training influence analysis
│
├── outliers/                     # Outlier detection
│   ├── knn_outlier.ipynb         # KNN-based outlier detection
│   └── sam_3_embeddings.ipynb    # SAM3 embedding analysis
│
├── evals/                        # Evaluation scripts
│   ├── eval_full_test.py         # Full test set evaluation
│   ├── eval_dangerous_test.py    # Dangerous subset eval
│   └── evaluate_convnext.py      # Classifier evaluation
│
├── export/                       # Model export
│   ├── export_yolo.py            # TensorRT export
│   └── deyo_export.py            # DEYO export
│
├── DEYO/                         # RT-DETR person detector
│   └── ultralytics/              # Custom ultralytics fork
│
├── notebooks/                    # Research notebooks
│   ├── attention_viz.ipynb       # Attention visualization
│   ├── manifold.ipynb            # Embedding manifold analysis
│   └── rf_detr.ipynb             # RF-DETR experiments
│
└── docs/                         # Documentation
    ├── PERSON_WEAPON.md          # Pipeline architecture
    └── temporal_filtering_experiments.md
```

---

## ⚡ Performance Benchmarks

### Inference Speed (RTX 4090)

| Precision | Latency/Frame | FPS | Use Case |
|-----------|--------------|-----|----------|
| FP32 | ~30-40ms | 25-33 | Maximum accuracy |
| FP16 | ~15-25ms | 40-65 | **Default (recommended)** |
| INT8 | ~10-20ms | 50-100 | Edge deployment |

### Temporal Filtering Impact

| Memory Buffer | Inference Time | Dropped Detections |
|--------------|----------------|-------------------|
| 1 second | 18.1ms | 0.3% |
| 2 seconds | 17.5ms | 0.3% |
| 4 seconds | 17.8ms | 0.2% |

**Recommended settings:**
```bash
--track --track_persist 120 --min_hits 5
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | `git lfs pull` |
| GPU OOM during training | Reduce batch size, use gradient accumulation |
| False positives | Use person+weapon pipeline, increase `--min_hits` |
| Slow inference | Use INT8 TensorRT engine, increase `--downscale` |
| Module conflicts (DEYO/YOLO) | Subprocess architecture handles this automatically |

---

## 📚 Technical References

- **YOLO11**: Ultralytics latest detection architecture
- **DEYO/RT-DETR**: Real-Time Detection Transformer (end-to-end, no NMS)
- **RF-DETR**: Roboflow Detection Transformer
- **ConvNeXTv2**: Facebook's modernized ConvNet
- **cleanlab**: Confident learning for label error detection
- **SAM3**: Segment Anything Model for embeddings
- **ByteTrack**: Simple and effective multi-object tracking
- **Captum**: PyTorch model interpretability library

---

## 📄 Requirements

- Python 3.12+
- CUDA GPU (16GB+ VRAM for training)
- TensorRT 8.6+ (for optimized inference)
- 20GB+ disk space for model exports

---

## 📞 Contact

For questions about implementation details or deployment assistance, please reach out.

---

<p align="center">
  <i>Built with ❤️ using SOTA deep learning techniques</i>
</p>
