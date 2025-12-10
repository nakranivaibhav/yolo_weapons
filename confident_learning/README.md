# Confident Learning for Label Error Detection

Find and fix wrong labels in your dataset using 5-fold cross-validation.

## How It Works

1. Split data into 5 folds
2. Train model on 4 folds, predict on the held-out fold
3. Repeat for all folds (every image gets a prediction from a model that never saw it)
4. Compare predictions vs labels → find mismatches

## Files

### 1. `prepare_data.ipynb`
**What:** Creates 5-fold splits from your dataset.

**Input:** `/workspace/yolo_dataset_4_dec`

**Output:** `/workspace/cv_folds_5fold/`
- `fold_0/` to `fold_4/` - each with train/val splits
- `fold_mapping.csv` - tracks which image is in which fold

**Run this first.**

---

### 2. `train_cv_folds.py`
**What:** Trains YOLO11n on each fold and collects predictions.

**Run:**
```bash
python train_cv_folds.py
```

**Output:** `/workspace/cv_folds_5fold/predictions/`
- `all_predictions.pkl` - predictions for all images
- `fold_X_predictions.json` - per-fold predictions

**Takes ~5-6 hours** (5 folds × 60-70 epochs each).

---

### 3. `prepare_matrix.ipynb`
**What:** Computes confident joint matrix and finds label errors.

**Input:** `all_predictions.pkl`

**Output:** 
- `label_errors.csv` - list of images with wrong labels
- `label_errors_inspection/` - folders organized by error type for visual review

**Run this last.**

---

## Quick Start

```bash
# Step 1: Prepare data (run notebook)
# Step 2: Train
python train_cv_folds.py

# Step 3: Analyze (run notebook)
```

## Output Folders for Review

After running `prepare_matrix.ipynb`, check:

```
/workspace/cv_folds_5fold/label_errors_inspection/
├── labeled_gun_pred_background/     ← gun label, model sees nothing
├── labeled_knife_pred_gun/          ← knife label, model sees gun
├── labeled_background_pred_rifle/   ← no label, model sees rifle (missed!)
└── ...
```

Each folder has images + label files for your team to review.
