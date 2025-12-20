import os
import cv2
from pathlib import Path
from collections import defaultdict

source_dataset = Path("/workspace/yolo_dataset_4_dec")
target_dataset = Path("/workspace/yolo_dataset_cls_cropped")

class_names = ['knife', 'gun', 'rifle', 'baseball_bat']
padding_percent = 0.20
min_crop_size = 32

splits = ['train', 'valid', 'test']

for split in splits:
    for class_name in class_names:
        target_dir = target_dataset / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

stats = defaultdict(lambda: defaultdict(int))
skipped = defaultdict(int)

for split in splits:
    images_dir = source_dataset / 'images' / split
    labels_dir = source_dataset / 'labels' / split
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist, skipping...")
        continue
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"\nProcessing {split} split ({len(image_files)} images)...")
    
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path.name}, skipping...")
            skipped[split] += 1
            continue
        
        img_height, img_width = img.shape[:2]
        
        if not label_path.exists():
            skipped[split] += 1
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            skipped[split] += 1
            continue
        
        crop_idx = 0
        for line in lines:
            parts = line.strip().split()
            if not parts or len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            pad_w = int(width_px * padding_percent)
            pad_h = int(height_px * padding_percent)
            
            x1_padded = max(0, x1 - pad_w)
            y1_padded = max(0, y1 - pad_h)
            x2_padded = min(img_width, x2 + pad_w)
            y2_padded = min(img_height, y2 + pad_h)
            
            crop_width = x2_padded - x1_padded
            crop_height = y2_padded - y1_padded
            
            if crop_width < min_crop_size or crop_height < min_crop_size:
                skipped[f"{split}_too_small"] += 1
                continue
            
            cropped_img = img[y1_padded:y2_padded, x1_padded:x2_padded]
            
            class_name = class_names[class_id]
            
            if crop_idx == 0:
                output_name = f"{image_path.stem}.jpg"
            else:
                output_name = f"{image_path.stem}_{crop_idx:03d}.jpg"
            
            output_path = target_dataset / split / class_name / output_name
            cv2.imwrite(str(output_path), cropped_img)
            
            stats[split][class_name] += 1
            crop_idx += 1

print("\n" + "="*60)
print("Cropped Classification Dataset Creation Complete!")
print("="*60)

for split in splits:
    print(f"\n{split.upper()}:")
    for class_name in class_names:
        count = stats[split][class_name]
        print(f"  {class_name}: {count} cropped images")
    if skipped[split] > 0:
        print(f"  Skipped (no labels): {skipped[split]}")
    if skipped[f"{split}_too_small"] > 0:
        print(f"  Skipped (too small): {skipped[f'{split}_too_small']}")

data_yaml_content = f"""path: yolo_dataset_cls_cropped
train: train
val: valid
test: test

nc: {len(class_names)}
names: {class_names}
"""

with open(target_dataset / 'data.yaml', 'w') as f:
    f.write(data_yaml_content)

print(f"\nCreated data.yaml at {target_dataset / 'data.yaml'}")
print(f"\nCropped classification dataset created at: {target_dataset}")
print(f"Padding used: {padding_percent*100}%")
