import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
from ultralytics.models.yolo import YOLO

CV_FOLDS_PATH = Path("/workspace/cv_folds_5fold")
OUTPUT_PATH = Path("/workspace/cv_folds_5fold/predictions_12_Dec")
START_FOLD = 0
NUM_FOLDS = 5
EPOCHS = 50  # Round 2: cleaner data converges faster
MODEL_NAME = "yolo11n.pt"
BATCH_SIZE = 32
IMGSZ = 640
DEVICE = 0
CONF_THRESHOLD = 0.01

CLASS_NAMES = ['knife', 'gun', 'rifle', 'baseball_bat']

custom_transforms = [
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
]

def parse_yolo_label(label_path):
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    boxes.append({
                        'class_id': cls_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    return boxes

def get_predictions_for_fold(model, fold_dir, conf_threshold=0.01):
    val_images_dir = fold_dir / 'images' / 'val'
    val_labels_dir = fold_dir / 'labels' / 'val'
    
    image_paths = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    
    predictions = {}
    
    for img_path in tqdm(image_paths, desc="Running inference"):
        img_name = img_path.stem
        label_path = val_labels_dir / f"{img_name}.txt"
        
        gt_boxes = parse_yolo_label(label_path)
        
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=0.5,
            verbose=False,
            device=DEVICE
        )
        
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxyn = boxes.xyxyn[i].cpu().numpy()
                
                x_center = (xyxyn[0] + xyxyn[2]) / 2
                y_center = (xyxyn[1] + xyxyn[3]) / 2
                width = xyxyn[2] - xyxyn[0]
                height = xyxyn[3] - xyxyn[1]
                
                pred_boxes.append({
                    'class_id': cls_id,
                    'confidence': conf,
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height)
                })
        
        is_background = len(gt_boxes) == 0
        
        predictions[str(img_path)] = {
            'image_path': str(img_path),
            'image_name': img_name,
            'ground_truth': gt_boxes,
            'predictions': pred_boxes,
            'is_background': is_background,
            'num_gt_boxes': len(gt_boxes),
            'num_pred_boxes': len(pred_boxes)
        }
    
    return predictions

def train_fold(fold_idx):
    fold_dir = CV_FOLDS_PATH / f"fold_{fold_idx}"
    data_yaml = fold_dir / "data.yaml"
    
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*60}")
    print(f"Data: {data_yaml}")
    
    model = YOLO(MODEL_NAME)
    
    results = model.train(
        val=False,
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=20,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        augmentations=custom_transforms,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        project=str(CV_FOLDS_PATH / 'runs'),
        name=f'fold_{fold_idx}',
        device=DEVICE,
        plots=True,
        exist_ok=True
    )
    
    model_path = CV_FOLDS_PATH / 'runs' / f'fold_{fold_idx}' / 'weights' / 'last.pt'
    print(f"\nModel saved to: {model_path}")
    
    return model_path

def collect_predictions(fold_idx, model_path):
    print(f"\n{'='*60}")
    print(f"COLLECTING PREDICTIONS FOR FOLD {fold_idx}")
    print(f"{'='*60}")
    
    fold_dir = CV_FOLDS_PATH / f"fold_{fold_idx}"
    
    model = YOLO(str(model_path))
    
    predictions = get_predictions_for_fold(model, fold_dir, CONF_THRESHOLD)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    pred_file = OUTPUT_PATH / f"fold_{fold_idx}_predictions.pkl"
    with open(pred_file, 'wb') as f:
        pickle.dump(predictions, f)
    
    pred_json = OUTPUT_PATH / f"fold_{fold_idx}_predictions.json"
    with open(pred_json, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to: {pred_file}")
    print(f"Total images: {len(predictions)}")
    print(f"Background images: {sum(1 for p in predictions.values() if p['is_background'])}")
    print(f"Images with objects: {sum(1 for p in predictions.values() if not p['is_background'])}")
    
    return predictions

def merge_all_predictions():
    print(f"\n{'='*60}")
    print("MERGING ALL FOLD PREDICTIONS")
    print(f"{'='*60}")
    
    all_predictions = {}
    
    for fold_idx in range(NUM_FOLDS):
        pred_file = OUTPUT_PATH / f"fold_{fold_idx}_predictions.pkl"
        with open(pred_file, 'rb') as f:
            fold_preds = pickle.load(f)
        
        for img_path, pred_data in fold_preds.items():
            pred_data['fold'] = fold_idx
            all_predictions[img_path] = pred_data
    
    merged_file = OUTPUT_PATH / "all_predictions.pkl"
    with open(merged_file, 'wb') as f:
        pickle.dump(all_predictions, f)
    
    rows = []
    for img_path, pred_data in all_predictions.items():
        gt_classes = [b['class_id'] for b in pred_data['ground_truth']]
        pred_classes = [b['class_id'] for b in pred_data['predictions']]
        pred_confs = [b['confidence'] for b in pred_data['predictions']]
        
        rows.append({
            'image_path': img_path,
            'image_name': pred_data['image_name'],
            'fold': pred_data['fold'],
            'is_background': pred_data['is_background'],
            'num_gt_boxes': pred_data['num_gt_boxes'],
            'num_pred_boxes': pred_data['num_pred_boxes'],
            'gt_classes': gt_classes,
            'pred_classes': pred_classes,
            'pred_confidences': pred_confs,
            'max_pred_conf': max(pred_confs) if pred_confs else 0.0
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH / "all_predictions.csv", index=False)
    df.to_pickle(OUTPUT_PATH / "all_predictions_df.pkl")
    
    print(f"\nMerged predictions saved to: {merged_file}")
    print(f"DataFrame saved to: {OUTPUT_PATH / 'all_predictions.csv'}")
    print(f"\nTotal images: {len(all_predictions)}")
    print(f"Background images: {df['is_background'].sum()}")
    print(f"Images with objects: {(~df['is_background']).sum()}")
    
    return all_predictions, df

def main():
    print(f"\n{'#'*60}")
    print("5-FOLD CV TRAINING FOR CONFIDENT LEARNING")
    print(f"{'#'*60}")
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Start Fold: {START_FOLD}")
    print(f"Folds: {NUM_FOLDS}")
    print(f"CV Folds Path: {CV_FOLDS_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    
    for fold_idx in range(START_FOLD, NUM_FOLDS):
        model_path = train_fold(fold_idx)
        collect_predictions(fold_idx, model_path)
    
    all_predictions, df = merge_all_predictions()
    
    print(f"\n{'#'*60}")
    print("TRAINING COMPLETE")
    print(f"{'#'*60}")
    print(f"\nAll predictions saved to: {OUTPUT_PATH}")
    print(f"\nNext step: Run confident learning analysis using:")
    print(f"  - {OUTPUT_PATH / 'all_predictions.pkl'}")
    print(f"  - {OUTPUT_PATH / 'all_predictions_df.pkl'}")

if __name__ == "__main__":
    main()
