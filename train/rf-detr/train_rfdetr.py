import os
import json
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from rfdetr import RFDETRNano
import argparse

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
    import shutil
    
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

def train_rfdetr(
    dataset_dir,
    output_dir,
    yolo_dataset_path=None,
    epochs=250,
    batch_size=64,
    warmup_epochs=2,
    resolution=640,
    weight_decay=3e-4,
    dropout=0.05,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if yolo_dataset_path:
        print("="*80)
        print("Converting YOLO dataset to COCO format...")
        print("="*80)
        yolo_to_coco(yolo_dataset_path, dataset_dir)
        copy_images_to_coco(yolo_dataset_path, dataset_dir)
        print("\n✅ Dataset conversion complete!")
    
    print("\n" + "="*80)
    print("Training RF-DETR Nano Model")
    print("="*80)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {resolution}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("="*80 + "\n")
    
    model = RFDETRNano(resolution=resolution)
    
    history = []
    
    def epoch_callback(data):
        history.append(data)
        if 'epoch' in data:
            print(f"\nEpoch {data['epoch']} completed")
            if 'loss' in data:
                print(f"Loss: {data['loss']:.4f}")
    
    model.callbacks["on_fit_epoch_end"].append(epoch_callback)
    
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        dropout=dropout,
    )
    
    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    print(f"Total epochs trained: {len(history)}")
    print(f"Model saved to: {output_dir}")
    
    if history:
        print("\nLast 5 epochs:")
        for epoch_data in history[-5:]:
            epoch_num = epoch_data.get('epoch', 'N/A')
            loss_val = epoch_data.get('loss', 'N/A')
            print(f"  Epoch {epoch_num}: Loss = {loss_val}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR model on dangerous weapons dataset')
    
    parser.add_argument('--yolo-dataset', type=str, default='/workspace/yolo_dataset_4_dec',
                        help='Path to YOLO format dataset (will be converted to COCO)')
    parser.add_argument('--coco-dataset', type=str, default='/workspace/coco_dataset',
                        help='Path to COCO format dataset (output of conversion)')
    parser.add_argument('--output-dir', type=str, default='/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano',
                        help='Output directory for trained model')
    parser.add_argument('--skip-conversion', action='store_true',
                        help='Skip YOLO to COCO conversion (use existing COCO dataset)')
    
    parser.add_argument('--resolution', type=int, default=640,
                        help='Input resolution')
    parser.add_argument('--weight-decay', type=float, default=3e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size per GPU')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs')
    
    args = parser.parse_args()
    
    yolo_dataset = args.yolo_dataset if not args.skip_conversion else None
    
    model, history = train_rfdetr(
        dataset_dir=args.coco_dataset,
        output_dir=args.output_dir,
        yolo_dataset_path=yolo_dataset,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
    )

if __name__ == '__main__':
    main()

