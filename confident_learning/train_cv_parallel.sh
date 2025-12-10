#!/bin/bash
set -e

cd /workspace/yolo_dangerous_weapons/confident_learning

echo "=============================================="
echo "5-FOLD CV PARALLEL TRAINING"
echo "Running 2 folds at a time (safe for 24GB VRAM)"
echo "=============================================="
echo ""

PYTHON_CMD="python -c"
TRAIN_SCRIPT='
import sys
from train_cv_folds import train_fold, collect_predictions
fold_idx = int(sys.argv[1])
print(f"Starting fold {fold_idx}...")
model_path = train_fold(fold_idx)
collect_predictions(fold_idx, model_path)
print(f"Fold {fold_idx} complete!")
'

run_fold() {
    local fold=$1
    echo "[$(date '+%H:%M:%S')] Starting fold $fold..."
    python -c "$TRAIN_SCRIPT" $fold 2>&1 | tee -a "fold_${fold}.log"
    echo "[$(date '+%H:%M:%S')] Fold $fold complete!"
}

echo "[$(date '+%H:%M:%S')] === BATCH 1: Folds 0 and 1 ==="
run_fold 0 &
PID0=$!
run_fold 1 &
PID1=$!

echo "Waiting for folds 0 and 1 (PIDs: $PID0, $PID1)..."
wait $PID0 $PID1
echo ""

echo "[$(date '+%H:%M:%S')] === BATCH 2: Folds 2 and 3 ==="
run_fold 2 &
PID2=$!
run_fold 3 &
PID3=$!

echo "Waiting for folds 2 and 3 (PIDs: $PID2, $PID3)..."
wait $PID2 $PID3
echo ""

echo "[$(date '+%H:%M:%S')] === BATCH 3: Fold 4 ==="
run_fold 4
echo ""

echo "[$(date '+%H:%M:%S')] === MERGING ALL PREDICTIONS ==="
python -c "from train_cv_folds import merge_all_predictions; merge_all_predictions()"

echo ""
echo "=============================================="
echo "ALL FOLDS COMPLETE!"
echo "=============================================="
echo "Predictions saved to: /workspace/cv_folds_5fold/predictions/"
echo "Log files: fold_0.log, fold_1.log, ..., fold_4.log"
