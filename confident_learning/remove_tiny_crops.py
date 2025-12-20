import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

FOLDS_PATH = Path('/workspace/yolo_dataset_cls_5fold')
MIN_WIDTH = 32
MIN_HEIGHT = 32
MIN_AREA = 1024  # 32x32

def check_and_remove_tiny_images():
    """Find and remove tiny crops from all folds."""
    
    removed = []
    kept = 0
    
    print(f"\n{'#'*60}")
    print("REMOVING TINY CROPS")
    print(f"{'#'*60}")
    print(f"Min width: {MIN_WIDTH}px")
    print(f"Min height: {MIN_HEIGHT}px")
    print(f"Min area: {MIN_AREA}px²")
    
    for fold_idx in range(5):
        fold_dir = FOLDS_PATH / f'fold_{fold_idx}'
        
        if not fold_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}")
        print(f"{'='*60}")
        
        for split in ['train', 'val']:
            split_dir = fold_dir / split
            
            if not split_dir.exists():
                continue
            
            all_images = []
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    all_images.extend(list(class_dir.glob('*.jpg')))
                    all_images.extend(list(class_dir.glob('*.png')))
            
            for img_path in tqdm(all_images, desc=f"Fold {fold_idx}/{split}"):
                img = cv2.imread(str(img_path))
                
                if img is None:
                    removed.append({
                        'path': str(img_path),
                        'reason': 'failed_to_read',
                        'width': None,
                        'height': None,
                        'area': None
                    })
                    img_path.unlink()
                    continue
                
                h, w = img.shape[:2]
                area = w * h
                
                if w < MIN_WIDTH or h < MIN_HEIGHT or area < MIN_AREA:
                    removed.append({
                        'path': str(img_path),
                        'reason': 'too_small',
                        'width': w,
                        'height': h,
                        'area': area
                    })
                    img_path.unlink()
                else:
                    kept += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Images kept: {kept}")
    print(f"Images removed: {len(removed)}")
    
    if removed:
        # Save report
        report_dir = FOLDS_PATH / 'cleanup_reports'
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'removed_tiny_crops_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'min_width': MIN_WIDTH,
                'min_height': MIN_HEIGHT,
                'min_area': MIN_AREA,
                'total_removed': len(removed),
                'total_kept': kept,
                'removed_images': removed
            }, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        # Size distribution of removed images
        print("\nRemoved images size distribution:")
        tiny = sum(1 for r in removed if r['width'] and r['width'] < 10)
        small = sum(1 for r in removed if r['width'] and 10 <= r['width'] < 20)
        medium = sum(1 for r in removed if r['width'] and 20 <= r['width'] < MIN_WIDTH)
        unreadable = sum(1 for r in removed if r['reason'] == 'failed_to_read')
        
        print(f"  < 10px wide: {tiny}")
        print(f"  10-20px wide: {small}")
        print(f"  20-{MIN_WIDTH}px wide: {medium}")
        print(f"  Unreadable: {unreadable}")
        
        # Show some examples
        print("\nExamples of removed images:")
        for r in removed[:10]:
            if r['width']:
                print(f"  {Path(r['path']).name}: {r['width']}x{r['height']}px")
            else:
                print(f"  {Path(r['path']).name}: {r['reason']}")

if __name__ == "__main__":
    check_and_remove_tiny_images()
    print(f"\n{'#'*60}")
    print("CLEANUP COMPLETE")
    print(f"{'#'*60}")


