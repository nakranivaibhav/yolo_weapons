import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ['knife', 'gun', 'rifle', 'baseball_bat']

def load_model(model_path, resolution=640, num_classes=4, device='cuda'):
    """Load RF-DETR model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'args' in checkpoint:
        args = checkpoint['args']
        num_classes = getattr(args, 'num_classes', num_classes)
        print(f"Detected {num_classes} classes from checkpoint")
    
    model = RFDETRNano(resolution=resolution)
    model.model.reinitialize_detection_head(num_classes)
    model.model.class_names = CLASS_NAMES
    
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
        print("Loading EMA weights")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Loading model weights")
    else:
        state_dict = checkpoint
    
    model.model.model.load_state_dict(state_dict, strict=True)
    model.model.model.to(device)
    model.model.model.eval()
    
    print("✅ Model loaded successfully")
    return model

def run_coco_evaluation(model, dataset_dir, split='test', threshold=0.001, device='cuda'):
    """
    Run full COCO evaluation on test set.
    """
    dataset_dir = Path(dataset_dir)
    ann_file = dataset_dir / split / '_annotations.coco.json'
    img_dir = dataset_dir / split
    
    if not ann_file.exists():
        print(f"❌ Annotation file not found: {ann_file}")
        return None
    
    print(f"\n{'='*80}")
    print(f"🔍 Running COCO Evaluation on {split} set")
    print(f"{'='*80}")
    print(f"Annotation file: {ann_file}")
    print(f"Image directory: {img_dir}")
    print(f"Confidence threshold: {threshold}")
    print(f"{'='*80}\n")
    
    coco_gt = COCO(str(ann_file))
    
    img_ids = coco_gt.getImgIds()
    print(f"Total images: {len(img_ids)}")
    
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    print(f"Categories: {[c['name'] for c in cats]}")
    
    results = []
    
    print(f"\nRunning inference on {len(img_ids)} images...")
    for img_id in tqdm(img_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            continue
        
        try:
            detections = model.predict(str(img_path), threshold=threshold)
            
            if detections and hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for box, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    results.append({
                        'image_id': img_id,
                        'category_id': int(cls_id),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(conf)
                    })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nTotal detections: {len(results)}")
    
    if len(results) == 0:
        print("❌ No detections found!")
        return None
    
    results_file = dataset_dir / f'{split}_predictions.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f"Predictions saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("📊 COCO Evaluation Results")
    print(f"{'='*80}\n")
    
    coco_dt = coco_gt.loadRes(str(results_file))
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
        'AR_1': coco_eval.stats[6],
        'AR_10': coco_eval.stats[7],
        'AR_100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }
    
    print(f"\n{'='*80}")
    print("📈 Summary")
    print(f"{'='*80}")
    print(f"  mAP @0.50:0.95  = {metrics['mAP']*100:.1f}%")
    print(f"  mAP @0.50       = {metrics['mAP_50']*100:.1f}%")
    print(f"  mAP @0.75       = {metrics['mAP_75']*100:.1f}%")
    print(f"  mAP (small)     = {metrics['mAP_small']*100:.1f}%")
    print(f"  mAP (medium)    = {metrics['mAP_medium']*100:.1f}%")
    print(f"  mAP (large)     = {metrics['mAP_large']*100:.1f}%")
    print(f"  AR @100         = {metrics['AR_100']*100:.1f}%")
    print(f"{'='*80}\n")
    
    print("\n📊 Per-Category AP:")
    for cat in cats:
        cat_id = cat['id']
        cat_name = cat['name']
        
        try:
            coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            coco_eval_cat.summarize()
            
            ap = coco_eval_cat.stats[0] if len(coco_eval_cat.stats) > 0 else 0.0
            ap50 = coco_eval_cat.stats[1] if len(coco_eval_cat.stats) > 1 else 0.0
            print(f"  {cat_name:15s}: AP={ap*100:.1f}%, AP50={ap50*100:.1f}%")
        except Exception as e:
            print(f"  {cat_name:15s}: Error - {e}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate RF-DETR on test set')
    
    parser.add_argument('--model-path', type=str, 
                        default='/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano_albu_07_Jan_2026/checkpoint_best_total.pth',
                        help='Path to RF-DETR checkpoint')
    parser.add_argument('--dataset-dir', type=str, 
                        default='/workspace/coco_dataset',
                        help='Path to COCO dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (test, valid)')
    parser.add_argument('--resolution', type=int, default=640,
                        help='Input resolution')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Confidence threshold for predictions (low for COCO eval P-R curves)')
    
    args = parser.parse_args()
    
    model = load_model(
        args.model_path,
        args.resolution,
        args.num_classes,
        args.device
    )
    
    metrics = run_coco_evaluation(
        model,
        args.dataset_dir,
        args.split,
        args.threshold,
        args.device
    )
    
    if metrics:
        print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
