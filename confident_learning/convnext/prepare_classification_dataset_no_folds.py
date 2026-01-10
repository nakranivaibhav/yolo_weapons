import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

YOLO_DATASET = Path('/workspace/yolo_dataset_4_dec')
CELLPHONE_DATASET = Path('/workspace/cell_phones_raw')
OUTPUT_PATH = Path('/workspace/yolo_dataset_cls_cropped')
EXPANSION = 0.3

CLASSES = ['knife', 'gun', 'rifle', 'baseball_bat', 'cell_phone', 'human']

def expand_bbox_yolo(x_center, y_center, w, h, expansion=0.3):
    """Expand YOLO bbox (normalized coordinates)."""
    w_expanded = w * (1 + expansion)
    h_expanded = h * (1 + expansion)
    return x_center, y_center, w_expanded, h_expanded

def expand_bbox_absolute(xmin, ymin, xmax, ymax, img_w, img_h, expansion=0.3):
    """Expand absolute pixel bbox."""
    w = xmax - xmin
    h = ymax - ymin
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    
    w_expanded = w * (1 + expansion)
    h_expanded = h * (1 + expansion)
    
    xmin_new = max(0, int(x_center - w_expanded / 2))
    ymin_new = max(0, int(y_center - h_expanded / 2))
    xmax_new = min(img_w, int(x_center + w_expanded / 2))
    ymax_new = min(img_h, int(y_center + h_expanded / 2))
    
    return xmin_new, ymin_new, xmax_new, ymax_new

def crop_yolo_bbox(img_path, bbox, expansion=0.3):
    """Crop image using YOLO bbox (normalized coordinates)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    x_center, y_center, box_w, box_h = expand_bbox_yolo(*bbox, expansion)
    
    xmin = int((x_center - box_w / 2) * w)
    ymin = int((y_center - box_h / 2) * h)
    xmax = int((x_center + box_w / 2) * w)
    ymax = int((y_center + box_h / 2) * h)
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    crop = img[ymin:ymax, xmin:xmax]
    return crop if crop.size > 0 else None

def crop_absolute_bbox(img_path, xmin, ymin, xmax, ymax):
    """Crop image using absolute pixel coordinates."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    crop = img[ymin:ymax, xmin:xmax]
    return crop if crop.size > 0 else None

def get_boxes_from_yolo_label(label_path):
    """Get all bounding boxes from YOLO label file."""
    if not label_path.exists():
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and len(parts) >= 5:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                boxes.append({
                    'class': cls_id,
                    'bbox': (x_center, y_center, w, h)
                })
    return boxes

def collect_weapon_crops_by_split():
    """Collect weapon crops from YOLO dataset, preserving train/valid/test splits."""
    print("\n" + "="*60)
    print("COLLECTING WEAPON CROPS FROM YOLO DATASET")
    print("="*60)
    
    crops_by_split = {'train': [], 'valid': [], 'test': []}
    
    for split in ['train', 'valid', 'test']:
        images_dir = YOLO_DATASET / 'images' / split
        labels_dir = YOLO_DATASET / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        for img_path in tqdm(list(images_dir.glob('*')), desc=f"Processing {split}"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            label_path = labels_dir / f"{img_path.stem}.txt"
            boxes = get_boxes_from_yolo_label(label_path)
            
            if not boxes:
                continue
            
            for i, box in enumerate(boxes):
                crop = crop_yolo_bbox(img_path, box['bbox'], EXPANSION)
                if crop is not None:
                    suffix = f"_crop{i}" if len(boxes) > 1 else ""
                    crops_by_split[split].append({
                        'crop': crop,
                        'class_idx': box['class'],
                        'filename': f"{img_path.stem}{suffix}{img_path.suffix}",
                        'split': split
                    })
    
    for split in ['train', 'valid', 'test']:
        print(f"  {split}: {len(crops_by_split[split])} crops")
    
    return crops_by_split

def collect_cellphone_crops():
    """Collect cell phone crops and assign to train split."""
    print("\n" + "="*60)
    print("COLLECTING CELL PHONE CROPS")
    print("="*60)
    
    labels_df = pd.read_csv(CELLPHONE_DATASET / 'labels.csv')
    crops = []
    
    cell_phone_idx = CLASSES.index('cell_phone')
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing cell phones"):
        img_path = CELLPHONE_DATASET / 'positive' / row['filename']
        
        if not img_path.exists():
            continue
        
        xmin, ymin, xmax, ymax = expand_bbox_absolute(
            row['xmin'], row['ymin'], row['xmax'], row['ymax'],
            row['width'], row['height'], EXPANSION
        )
        
        crop = crop_absolute_bbox(img_path, xmin, ymin, xmax, ymax)
        
        if crop is not None:
            crops.append({
                'crop': crop,
                'class_idx': cell_phone_idx,
                'filename': row['filename'],
                'split': 'train'
            })
    
    print(f"Collected {len(crops)} cell phone crops (assigned to train)")
    return crops

def collect_human_images_by_split():
    """Collect crowdhuman full images (hard negatives), preserving splits."""
    print("\n" + "="*60)
    print("COLLECTING HUMAN IMAGES (FULL IMAGES)")
    print("="*60)
    
    images_by_split = {'train': [], 'valid': [], 'test': []}
    human_idx = CLASSES.index('human')
    
    for split in ['train', 'valid', 'test']:
        images_dir = YOLO_DATASET / 'images' / split
        
        if not images_dir.exists():
            continue
        
        for img_path in tqdm(list(images_dir.glob('crowdhuman_*')), desc=f"Processing {split}"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            img = cv2.imread(str(img_path))
            if img is not None:
                images_by_split[split].append({
                    'crop': img,
                    'class_idx': human_idx,
                    'filename': img_path.name,
                    'split': split
                })
    
    for split in ['train', 'valid', 'test']:
        print(f"  {split}: {len(images_by_split[split])} images")
    
    return images_by_split

def save_single_image(args):
    """Save a single crop image (for parallel processing)."""
    crop_data, class_name, dst_path = args
    try:
        cv2.imwrite(str(dst_path), crop_data['crop'])
        return True
    except Exception as e:
        print(f"Error writing {dst_path}: {e}")
        return False

def save_split_structure(all_crops_by_split, max_workers=16):
    """Save crops to train/valid/test structure with parallel writes."""
    print(f"\n" + "="*60)
    print("SAVING DATASET STRUCTURE")
    print("="*60)
    
    # Create directories
    for split in ['train', 'valid', 'test']:
        for class_name in CLASSES:
            (OUTPUT_PATH / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Prepare all write tasks
    write_tasks = []
    for split in ['train', 'valid', 'test']:
        print(f"\nPreparing {split} split...")
        for crop_data in all_crops_by_split[split]:
            class_name = CLASSES[crop_data['class_idx']]
            dst_path = OUTPUT_PATH / split / class_name / crop_data['filename']
            write_tasks.append((crop_data, class_name, dst_path))
    
    # Parallel write with progress bar
    print(f"\nWriting {len(write_tasks)} images...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(save_single_image, write_tasks),
            total=len(write_tasks),
            desc="Writing images"
        ))

def main():
    print("\n" + "#"*60)
    print("PREPARING CLASSIFICATION DATASET (NO FOLDS)")
    print("#"*60)
    print(f"YOLO Dataset: {YOLO_DATASET}")
    print(f"Cell Phone Dataset: {CELLPHONE_DATASET}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Expansion: {EXPANSION * 100}%")
    print(f"Classes: {CLASSES}")
    
    # Collect all data by split
    weapon_crops_by_split = collect_weapon_crops_by_split()
    cellphone_crops = collect_cellphone_crops()
    human_images_by_split = collect_human_images_by_split()
    
    # Combine data maintaining splits
    all_crops_by_split = {
        'train': weapon_crops_by_split['train'] + cellphone_crops + human_images_by_split['train'],
        'valid': weapon_crops_by_split['valid'] + human_images_by_split['valid'],
        'test': weapon_crops_by_split['test'] + human_images_by_split['test']
    }
    
    print(f"\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for split in ['train', 'valid', 'test']:
        print(f"\n{split.upper()} Split: {len(all_crops_by_split[split])} samples")
        split_labels = [c['class_idx'] for c in all_crops_by_split[split]]
        for idx, class_name in enumerate(CLASSES):
            count = split_labels.count(idx)
            print(f"  {class_name}: {count}")
    
    total_samples = sum(len(all_crops_by_split[s]) for s in ['train', 'valid', 'test'])
    print(f"\nTotal samples: {total_samples}")
    
    # Save structure
    save_split_structure(all_crops_by_split, max_workers=16)
    
    print(f"\n" + "#"*60)
    print("COMPLETE!")
    print("#"*60)
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"\nStructure:")
    print(f"  train/ valid/ test/")
    print(f"    {', '.join(CLASSES)}/")

if __name__ == "__main__":
    main()

