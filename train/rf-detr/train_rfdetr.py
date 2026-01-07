import argparse
from pathlib import Path
from rfdetr import RFDETRNano

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
    lr_scheduler='cosine',
    lr_min_factor=0.03,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
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
    print(f"  Multi-scale: {multi_scale}")
    print(f"  Expanded scales: {expanded_scales}")
    print(f"  Random resize via padding: {do_random_resize_via_padding}")
    print(f"  EMA: {use_ema} (decay: {ema_decay})")
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
    parser = argparse.ArgumentParser(description='Train RF-DETR model on dangerous weapons dataset')
    
    parser.add_argument('--coco-dataset', type=str, default='/workspace/coco_dataset',
                        help='Path to COCO format dataset')
    parser.add_argument('--output-dir', type=str, default='/workspace/yolo_dangerous_weapons/models/rfdetr/dangerous_weapons_nano',
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
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        do_random_resize_via_padding=args.random_resize_padding,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )

if __name__ == '__main__':
    main()
