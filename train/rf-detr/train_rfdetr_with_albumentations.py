import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import albumentations as A
from rfdetr import RFDETRNano
import rfdetr.datasets.coco as rfdetr_coco
import rfdetr.datasets.transforms as T


class AlbumentationsTransform:
    """
    Wrapper to apply Albumentations augmentations within RF-DETR's transform pipeline.
    Works with (PIL Image, target_dict) pairs where target contains 'boxes' in xyxy format.
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img, target):
        img_np = np.array(img)
        
        if 'boxes' not in target or len(target['boxes']) == 0:
            transformed = self.transform(image=img_np, bboxes=[], category_ids=[])
            return Image.fromarray(transformed['image']), target
        
        boxes = target['boxes'].numpy()
        h, w = img_np.shape[:2]
        
        bboxes_albu = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                bboxes_albu.append([x1, y1, x2, y2])
        
        labels = target['labels'].numpy().tolist()
        
        if len(bboxes_albu) < len(labels):
            labels = labels[:len(bboxes_albu)]
        
        if len(bboxes_albu) == 0:
            transformed = self.transform(image=img_np, bboxes=[], category_ids=[])
            return Image.fromarray(transformed['image']), target
        
        try:
            transformed = self.transform(
                image=img_np,
                bboxes=bboxes_albu,
                category_ids=labels
            )
        except Exception as e:
            return img, target
        
        aug_img = Image.fromarray(transformed['image'])
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['category_ids']
        
        if len(aug_bboxes) > 0:
            target = target.copy()
            target['boxes'] = torch.tensor(aug_bboxes, dtype=torch.float32)
            target['labels'] = torch.tensor(aug_labels, dtype=torch.int64)
            
            areas = (target['boxes'][:, 2] - target['boxes'][:, 0]) * \
                    (target['boxes'][:, 3] - target['boxes'][:, 1])
            target['area'] = areas
            target['iscrowd'] = torch.zeros(len(aug_bboxes), dtype=torch.int64)
        
        return aug_img, target


def get_albumentations_transform():
    return A.Compose([
        A.MotionBlur(blur_limit=(3, 7), p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.15),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.1),
        A.ImageCompression(quality_range=(50, 95), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0.5, 1, 1), p=0.15),
        A.ToGray(p=0.05),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.15),
        A.CLAHE(clip_limit=2.0, p=0.15),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['category_ids'],
        min_visibility=0.3,
        min_area=100
    ))


def patch_rfdetr_transforms():
    """
    Monkey-patch RF-DETR's transform functions to include Albumentations.
    """
    original_make_transforms = rfdetr_coco.make_coco_transforms_square_div_64
    original_make_transforms_regular = rfdetr_coco.make_coco_transforms
    
    albu_transform = get_albumentations_transform()
    albu_wrapper = AlbumentationsTransform(albu_transform)
    
    def patched_make_transforms_square(image_set, resolution, multi_scale=False, 
                                       expanded_scales=False, skip_random_resize=False, 
                                       patch_size=16, num_windows=4):
        original = original_make_transforms(
            image_set, resolution, multi_scale, expanded_scales, 
            skip_random_resize, patch_size, num_windows
        )
        
        if image_set == 'train':
            return T.Compose([albu_wrapper] + original.transforms)
        return original
    
    def patched_make_transforms_regular(image_set, resolution, multi_scale=False,
                                        expanded_scales=False, skip_random_resize=False,
                                        patch_size=16, num_windows=4):
        original = original_make_transforms_regular(
            image_set, resolution, multi_scale, expanded_scales,
            skip_random_resize, patch_size, num_windows
        )
        
        if image_set == 'train':
            return T.Compose([albu_wrapper] + original.transforms)
        return original
    
    rfdetr_coco.make_coco_transforms_square_div_64 = patched_make_transforms_square
    rfdetr_coco.make_coco_transforms = patched_make_transforms_regular
    
    print("✅ Albumentations transforms patched into RF-DETR pipeline")
    print("   Applied augmentations:")
    print("   - MotionBlur, GaussianBlur")
    print("   - GaussNoise, ISONoise")
    print("   - ImageCompression")
    print("   - RandomBrightnessContrast, RandomGamma")
    print("   - HueSaturationValue")
    print("   - RandomShadow")
    print("   - ToGray, Sharpen, CLAHE")


def train_rfdetr(
    dataset_dir,
    output_dir,
    epochs=250,
    batch_size=8,
    warmup_epochs=2,
    resolution=640,
    weight_decay=3e-4,
    dropout=0.05,
    device='cuda',
    multi_scale=True,
    expanded_scales=True,
    do_random_resize_via_padding=True,
    use_ema=True,
    ema_decay=0.9998,
    lr=1.3e-4,
    lr_encoder=8e-5,
    grad_accum_steps=2,
    num_workers=2,
    use_albumentations=False,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_albumentations:
        patch_rfdetr_transforms()
    
    print("\n" + "="*80)
    print("Training RF-DETR Nano Model")
    print("="*80)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {resolution}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"Gradient Accumulation: {grad_accum_steps}")
    print(f"Learning Rate: {lr}")
    print(f"Encoder LR: {lr_encoder}")
    print(f"Warmup Epochs: {warmup_epochs}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Dropout: {dropout}")
    print(f"Device: {device}")
    print(f"\nAugmentations:")
    print(f"  Albumentations (on-the-fly): {use_albumentations}")
    print(f"  Multi-scale: {multi_scale}")
    print(f"  Expanded scales: {expanded_scales}")
    print(f"  Random resize via padding: {do_random_resize_via_padding}")
    print(f"  EMA: {use_ema} (decay: {ema_decay})")
    print(f"\nDataloader:")
    print(f"  Num workers: {num_workers}")
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
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        lr_encoder=lr_encoder,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        dropout=dropout,
        output_dir=str(output_dir),
        device=device,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        use_ema=use_ema,
        ema_decay=ema_decay,
        num_workers=num_workers,
        tensorboard=True,
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
    parser = argparse.ArgumentParser(description='Train RF-DETR with Albumentations (on-the-fly)')
    
    parser.add_argument('--coco-dataset', type=str, default='/workspace/coco_dataset',
                        help='Path to COCO format dataset')
    parser.add_argument('--output-dir', type=str, default='/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano_albu',
                        help='Output directory for trained model')
    
    parser.add_argument('--resolution', type=int, default=640,
                        help='Input resolution')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--grad-accum-steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1.3e-4,
                        help='Learning rate')
    parser.add_argument('--lr-encoder', type=float, default=8e-5,
                        help='Encoder learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=3e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: "cuda", "cpu", or "mps"')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of dataloader workers')
    
    parser.add_argument('--use-albumentations', action='store_true',
                        help='Enable Albumentations (on-the-fly)')
    parser.add_argument('--multi-scale', action='store_true', default=True,
                        help='Enable multi-scale training')
    parser.add_argument('--no-multi-scale', action='store_false', dest='multi_scale',
                        help='Disable multi-scale training')
    parser.add_argument('--expanded-scales', action='store_true', default=True,
                        help='Enable expanded scale augmentation')
    parser.add_argument('--no-expanded-scales', action='store_false', dest='expanded_scales',
                        help='Disable expanded scale augmentation')
    parser.add_argument('--random-resize-padding', action='store_true', default=True,
                        help='Enable random resize via padding')
    parser.add_argument('--no-random-resize-padding', action='store_false', dest='random_resize_padding',
                        help='Disable random resize via padding')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='Enable EMA')
    parser.add_argument('--no-ema', action='store_false', dest='use_ema',
                        help='Disable EMA')
    parser.add_argument('--ema-decay', type=float, default=0.9998,
                        help='EMA decay rate')
    
    args = parser.parse_args()
    
    model, history = train_rfdetr(
        dataset_dir=args.coco_dataset,
        output_dir=args.output_dir,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        device=args.device,
        num_workers=args.num_workers,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        do_random_resize_via_padding=args.random_resize_padding,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        use_albumentations=args.use_albumentations,
    )


if __name__ == '__main__':
    main()
