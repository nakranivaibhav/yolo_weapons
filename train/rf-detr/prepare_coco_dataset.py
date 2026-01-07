import json
import yaml
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def yolo_to_coco(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_path / 'data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    categories = [
        {'id': i, 'name': name, 'supercategory': 'object'} 
        for i, name in enumerate(data_config['names'])
    ]
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            print(f"Skipping {split} - directory not found")
            continue
        
        split_output_dir = output_path / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': categories
        }
        
        image_id = 0
        annotation_id = 0
        
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
        
        for img_path in tqdm(image_files, desc=f"Converting {split}"):
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': img_path.name,
                    'width': width,
                    'height': height
                })
                
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            x_min = (x_center - w / 2) * width
                            y_min = (y_center - h / 2) * height
                            bbox_width = w * width
                            bbox_height = h * height
                            
                            coco_data['annotations'].append({
                                'id': annotation_id,
                                'image_id': image_id,
                                'category_id': class_id,
                                'bbox': [x_min, y_min, bbox_width, bbox_height],
                                'area': bbox_width * bbox_height,
                                'iscrowd': 0
                            })
                            annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        output_file = split_output_dir / "_annotations.coco.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        print(f"Output: {output_file}")

def copy_images_to_coco(source_path, dest_path):
    print("\nCopying images to COCO dataset structure...")
    
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    
    for split in ['train', 'valid', 'test']:
        source_images = source_path / 'images' / split
        dest_images = dest_path / split
        
        if source_images.exists():
            dest_images.mkdir(parents=True, exist_ok=True)
            print(f"\nCopying {split} images...")
            
            for img_file in tqdm(list(source_images.glob('*')), desc=f"Copying {split}"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_file = dest_images / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)
            
            image_count = len([f for f in dest_images.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"Total {split} images: {image_count}")

def verify_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    
    print("\n" + "="*80)
    print("Dataset Verification")
    print("="*80)
    
    total_images = 0
    total_annotations = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"⚠️  {split}: directory not found")
            continue
        
        anno_file = split_dir / "_annotations.coco.json"
        if not anno_file.exists():
            print(f"⚠️  {split}: annotations file not found")
            continue
        
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        num_images = len(coco_data['images'])
        num_annotations = len(coco_data['annotations'])
        num_categories = len(coco_data['categories'])
        
        total_images += num_images
        total_annotations += num_annotations
        
        print(f"\n{split}:")
        print(f"  Images: {num_images}")
        print(f"  Annotations: {num_annotations}")
        print(f"  Categories: {num_categories}")
        
        if num_categories > 0:
            print(f"  Classes: {[cat['name'] for cat in coco_data['categories']]}")
    
    print("\n" + "="*80)
    print(f"Total Images: {total_images}")
    print(f"Total Annotations: {total_annotations}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to COCO format for RF-DETR')
    
    parser.add_argument('--yolo-dataset', type=str, default='/workspace/yolo_dataset_4_dec',
                        help='Path to YOLO format dataset')
    parser.add_argument('--coco-dataset', type=str, default='/workspace/coco_dataset',
                        help='Path to output COCO format dataset')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing COCO dataset without conversion')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.coco_dataset)
    else:
        print("="*80)
        print("Converting YOLO dataset to COCO format")
        print("="*80)
        print(f"Source: {args.yolo_dataset}")
        print(f"Destination: {args.coco_dataset}")
        print("="*80)
        
        yolo_to_coco(args.yolo_dataset, args.coco_dataset)
        copy_images_to_coco(args.yolo_dataset, args.coco_dataset)
        
        print("\n✅ Dataset conversion complete!")
        
        verify_dataset(args.coco_dataset)

if __name__ == '__main__':
    main()

