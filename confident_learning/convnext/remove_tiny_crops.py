import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

FOLDS_PATH = Path('/workspace/yolo_dataset_cls_5fold')
MIN_WIDTH = 16
MIN_HEIGHT = 16
MIN_AREA = 256  # 16x16 - very small threshold for truly tiny crops

def extract_source_image(filename):
    """Extract source image information from crop filename.
    
    Filename format: {crop_id}_{image_id}_{split}_{dataset}_{original_name}
    Example: 0_3880_valid_dangerous_valid_000151.jpg
    """
    try:
        parts = filename.split('_')
        if len(parts) >= 4:
            crop_id = parts[0]
            image_id = parts[1]
            split = parts[2]
            dataset = parts[3]
            original = '_'.join(parts[4:]) if len(parts) > 4 else 'unknown'
            return f"{dataset}_{split}_{original}"
        return filename
    except:
        return filename

def check_and_remove_tiny_images():
    """Find and remove tiny crops from all folds."""
    
    removed = []
    kept = 0
    
    print(f"\n{'#'*60}")
    print("REMOVING TRULY TINY CROPS")
    print(f"{'#'*60}")
    print(f"Removal criteria: (width < {MIN_WIDTH} AND height < {MIN_HEIGHT}) OR area < {MIN_AREA}")
    print(f"Note: Elongated crops (e.g., 10x80px) will be KEPT")
    
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
                    source_img = extract_source_image(img_path.name)
                    removed.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'source_image': source_img,
                        'reason': 'failed_to_read',
                        'width': None,
                        'height': None,
                        'area': None
                    })
                    img_path.unlink()
                    continue
                
                h, w = img.shape[:2]
                area = w * h
                
                if (w < MIN_WIDTH and h < MIN_HEIGHT) or area < MIN_AREA:
                    source_img = extract_source_image(img_path.name)
                    removed.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'source_image': source_img,
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
        
        from collections import defaultdict
        by_source = defaultdict(list)
        for r in removed:
            by_source[r['source_image']].append(r)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'criteria': f"(width < {MIN_WIDTH} AND height < {MIN_HEIGHT}) OR area < {MIN_AREA}",
                'min_width': MIN_WIDTH,
                'min_height': MIN_HEIGHT,
                'min_area': MIN_AREA,
                'total_removed': len(removed),
                'total_kept': kept,
                'unique_source_images': len(by_source),
                'removed_images': removed,
                'by_source_image': {k: [r['filename'] for r in v] for k, v in by_source.items()}
            }, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        # Size distribution of removed images
        print("\nRemoved images size distribution:")
        tiny = sum(1 for r in removed if r['width'] and r['width'] < 10 and r['height'] < 10)
        small = sum(1 for r in removed if r['width'] and r['area'] and r['area'] < 500)
        medium = sum(1 for r in removed if r['width'] and r['area'] and 500 <= r['area'] < MIN_AREA)
        unreadable = sum(1 for r in removed if r['reason'] == 'failed_to_read')
        
        print(f"  < 10x10px: {tiny}")
        print(f"  Area < 500px²: {small}")
        print(f"  Area {MIN_AREA}px² > area >= 500px²: {medium}")
        print(f"  Unreadable: {unreadable}")
        
        from collections import defaultdict
        by_source_display = defaultdict(list)
        for r in removed:
            by_source_display[r['source_image']].append(r)
        
        print(f"\nRemoved crops from {len(by_source_display)} unique source images")
        print("\nTop 10 source images with most removed crops:")
        sorted_sources = sorted(by_source_display.items(), key=lambda x: len(x[1]), reverse=True)
        for src, crops in sorted_sources[:10]:
            print(f"  {src}: {len(crops)} crops removed")
        
        # Show some examples
        print("\nExamples of removed images:")
        for r in removed[:10]:
            if r['width']:
                print(f"  {r['filename']}: {r['width']}x{r['height']}px (area={r['area']}) from {r['source_image']}")
            else:
                print(f"  {r['filename']}: {r['reason']} from {r['source_image']}")

if __name__ == "__main__":
    check_and_remove_tiny_images()
    print(f"\n{'#'*60}")
    print("CLEANUP COMPLETE")
    print(f"{'#'*60}")


