# RF-DETR Training for Dangerous Weapons Detection

Complete training pipeline for RF-DETR models with advanced augmentation support.

## Quick Start

### 1. Prepare Dataset (One-time)

Convert YOLO dataset to COCO format:

```bash
cd /workspace/yolo_dangerous_weapons
./train/rf-detr/prepare_dataset.sh
```

### 2. Train Model

**Single GPU:**
```bash
./train/rf-detr/train_rfdetr.sh
```

**Multi-GPU (2 GPUs):**
```bash
./train/rf-detr/train_rfdetr_multigpu.sh
```

## Augmentation Features

RF-DETR includes powerful built-in augmentations that are **enabled by default**:

### 1. Multi-Scale Training (`--multi-scale`)
- Trains on multiple image scales
- Helps model detect objects at various sizes
- **Default: Enabled**

### 2. Expanded Scales (`--expanded-scales`)
- Uses wider range of scale variations
- Improves robustness to scale changes
- **Default: Enabled**

### 3. Random Resize via Padding (`--random-resize-padding`)
- Randomly resizes images while maintaining aspect ratio
- Adds padding to reach target size
- Reduces distortion compared to direct resizing
- **Default: Enabled**

### 4. Exponential Moving Average (EMA) (`--use-ema`)
- Maintains moving average of model weights
- Improves final model stability and performance
- Decay rate: 0.9998
- **Default: Enabled**

### 5. Default COCO Augmentations
RF-DETR automatically applies:
- Random horizontal flipping
- Random cropping
- Color jittering (brightness, contrast, saturation)
- These are built into the COCO dataset loader

## Training Configuration

### Recommended Settings (Default)

```bash
Resolution: 640
Epochs: 50
Batch Size: 8 per GPU
Gradient Accumulation: 2 steps (effective batch = 16)
Learning Rate: 1.3e-4
Encoder LR: 8e-5
Warmup: 2 epochs
Weight Decay: 3e-4
Dropout: 0.05
EMA Decay: 0.9998
```

### Custom Training

```bash
python train/rf-detr/train_rfdetr.py \
    --coco-dataset /workspace/coco_dataset \
    --output-dir /workspace/yolo_dangerous_weapons/models/rfdetr/custom_run \
    --resolution 640 \
    --epochs 100 \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --lr 1.3e-4 \
    --lr-encoder 8e-5 \
    --multi-scale \
    --expanded-scales \
    --random-resize-padding \
    --use-ema \
    --ema-decay 0.9998
```

## Disabling Augmentations

If you want to train **without** certain augmentations:

```bash
python train/rf-detr/train_rfdetr.py \
    --no-multi-scale \           # Disable multi-scale
    --no-expanded-scales \       # Disable expanded scales
    --no-random-resize-padding \ # Disable random resize
    --no-ema                     # Disable EMA
```

## Command Line Arguments

### Dataset & Output
```
--coco-dataset PATH          COCO format dataset directory
--output-dir PATH            Output directory for trained model
```

### Model & Training
```
--resolution INT             Input resolution (default: 640)
--epochs INT                 Number of epochs (default: 250)
--batch-size INT             Batch size per GPU (default: 8)
--grad-accum-steps INT       Gradient accumulation (default: 2)
--device STR                 Device: "cuda", "cpu", "mps" (default: cuda)
```

### Learning Rate
```
--lr FLOAT                   Main learning rate (default: 1.3e-4)
--lr-encoder FLOAT           Encoder learning rate (default: 8e-5)
--warmup-epochs INT          Warmup epochs (default: 2)
```

### Regularization
```
--weight-decay FLOAT         Weight decay (default: 3e-4)
--dropout FLOAT              Dropout rate (default: 0.05)
```

### Augmentation Flags
```
--multi-scale               Enable multi-scale training (default)
--no-multi-scale            Disable multi-scale training

--expanded-scales           Enable expanded scales (default)
--no-expanded-scales        Disable expanded scales

--random-resize-padding     Enable random resize via padding (default)
--no-random-resize-padding  Disable random resize via padding

--use-ema                   Enable EMA (default)
--no-ema                    Disable EMA
--ema-decay FLOAT           EMA decay rate (default: 0.9998)
```

## Dataset Structure

### Input: YOLO Format
```
yolo_dataset_4_dec/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
└── data.yaml
```

### Output: COCO Format
```
coco_dataset/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── *.jpg
│   └── _annotations.coco.json
└── test/
    ├── *.jpg
    └── _annotations.coco.json
```

## Classes

4 dangerous weapon classes:
- `knife` (class 0)
- `gun` (class 1)
- `rifle` (class 2)
- `baseball_bat` (class 3)

## Output Structure

```
models/rfdetr/dangerous_weapons_nano_<date>/
├── best.pth              # Best model checkpoint
├── last.pth              # Last epoch checkpoint
├── config.yaml           # Training configuration
├── events.out.tfevents.* # TensorBoard logs
└── checkpoints/          # Intermediate checkpoints
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano_<date>
```

### Training Metrics

During training, you'll see COCO evaluation metrics:

```
Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.739  ← Main metric
Average Precision (AP) @[ IoU=0.50      ] = 0.951  ← Loose IoU
Average Precision (AP) @[ IoU=0.75      ] = 0.787  ← Strict IoU
Average Recall    (AR) @[ IoU=0.50:0.95 ] = 0.839  ← Detection rate
```

**Good performance indicators:**
- AP@0.50:0.95 > 0.70 (70%)
- AP@0.50 > 0.90 (90%)
- AR > 0.80 (80%)

## Hardware Requirements

### Single GPU
- GPU: 16GB+ VRAM (RTX 4090, A100, etc.)
- Batch size: 8
- Grad accumulation: 2
- Effective batch: 16

### Multi-GPU (2 GPUs)
- GPU: 2x 12GB+ VRAM
- Batch size: 4 per GPU
- Grad accumulation: 2
- Effective batch: 16 total

### Low Memory (<12GB)
```bash
python train/rf-detr/train_rfdetr.py \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --resolution 512
```

## Tips for Best Results

### 1. Use All Augmentations
Keep all augmentations enabled (default) for best generalization.

### 2. Proper Learning Rates
- Main LR: 1.3e-4
- Encoder LR: 8e-5 (lower for pretrained encoder)

### 3. EMA is Critical
EMA significantly improves final model quality. Keep it enabled.

### 4. Multi-Scale Training
Essential for detecting weapons at various distances.

### 5. Warmup
2-5 epochs warmup helps stabilize early training.

### 6. Gradient Accumulation
Use to increase effective batch size without OOM.

## Comparison with YOLO

| Feature | YOLO11 | RF-DETR Nano |
|---------|--------|--------------|
| Architecture | CNN | Transformer |
| mAP (our dataset) | ~70% | ~74% |
| Small objects | Good | Excellent |
| Training time | 2-3 hours | 8-12 hours |
| Inference speed | Very Fast | Fast |
| Parameters | ~25M | ~10M |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 2 --grad-accum-steps 4

# Reduce resolution
--resolution 512

# Disable gradient checkpointing if enabled
```

### Poor Small Object Detection
```bash
# Increase resolution
--resolution 896

# Ensure multi-scale is enabled
--multi-scale --expanded-scales
```

### Training Unstable
```bash
# Increase warmup
--warmup-epochs 5

# Reduce learning rate
--lr 1e-4 --lr-encoder 5e-5

# Increase weight decay
--weight-decay 5e-4
```

### Overfitting
```bash
# Increase dropout
--dropout 0.1

# Increase weight decay
--weight-decay 5e-4

# Ensure augmentations are enabled
--multi-scale --expanded-scales --random-resize-padding
```

## Files

- `prepare_coco_dataset.py` - Convert YOLO to COCO format
- `prepare_dataset.sh` - Dataset conversion wrapper
- `train_rfdetr.py` - Main training script
- `train_rfdetr.sh` - Single GPU training wrapper
- `train_rfdetr_multigpu.sh` - Multi-GPU training wrapper

## References

- RF-DETR GitHub: https://github.com/roboflow/rf-detr
- RF-DETR Paper: "RF-DETR: Real-time Detection Transformer"
- Roboflow Augmentation Guide: https://roboflow.com/augment/rf-detr

