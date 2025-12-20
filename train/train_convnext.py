import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, 
    Resize, CenterCrop, RandomRotation, ColorJitter, RandomGrayscale,
    GaussianBlur, RandomAdjustSharpness, RandomAffine
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXT for image classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, default="facebook/convnextv2-base-22k-224", help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./convnext_finetuned", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_steps", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every N epochs")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataset(data_dir):
    data_path = Path(data_dir)
    
    dataset_dict = {}
    if (data_path / "train").exists():
        dataset_dict["train"] = str(data_path / "train")
    if (data_path / "valid").exists():
        dataset_dict["validation"] = str(data_path / "valid")
    if (data_path / "test").exists():
        dataset_dict["test"] = str(data_path / "test")
    
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    labels = dataset["train"].features["label"].names
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}
    
    return dataset, id2label, label2id


def get_transforms(image_processor, is_train=True):
    if isinstance(image_processor.size, dict):
        size = image_processor.size.get("shortest_edge") or image_processor.size.get("height") or 224
    else:
        size = image_processor.size
    
    mean = image_processor.image_mean
    std = image_processor.image_std
    
    if is_train:
        transform = A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=(7, 25), p=1.0),
                A.Defocus(radius=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
            A.ImageCompression(quality_lower=30, quality_upper=90, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.Downscale(scale_min=0.25, scale_max=0.75, p=0.3),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0.5, 1, 1), p=0.2),
            
            A.RandomSunFlare(src_radius=50, num_flare_circles_lower=1, num_flare_circles_upper=3, p=0.15),
            A.RandomToneCurve(scale=0.2, p=0.2),
            
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.3),
            A.ToGray(p=0.1),
            
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(height=size, width=size),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    return transform


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predicted = logits.argmax(-1)
            correct += (predicted == batch["labels"]).sum().item()
            total += batch["labels"].shape[0]
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train(args):
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset, id2label, label2id = load_and_prepare_dataset(args.data_dir)
    print(f"Classes: {list(id2label.values())}")
    print(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Validation samples: {len(dataset['validation'])}")
    
    print(f"Loading image processor from {args.model_name}...")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    train_transform = get_transforms(image_processor, is_train=True)
    val_transform = get_transforms(image_processor, is_train=False)
    
    def train_transforms(examples):
        examples["pixel_values"] = [
            train_transform(image=np.array(image.convert("RGB")))["image"] 
            for image in examples["image"]
        ]
        return examples
    
    def val_transforms(examples):
        examples["pixel_values"] = [
            val_transform(image=np.array(image.convert("RGB")))["image"] 
            for image in examples["image"]
        ]
        return examples
    
    train_dataset = dataset["train"].with_transform(train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = None
    if "validation" in dataset:
        val_dataset = dataset["validation"].with_transform(val_transforms)
        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    print(f"Loading model from {args.model_name}...")
    model = ConvNextV2ForImageClassification.from_pretrained(
        args.model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
    
    print("\nStarting training...")
    best_val_accuracy = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = logits.argmax(-1)
            correct += (predicted == batch["labels"]).sum().item()
            total += batch["labels"].shape[0]
            
            global_step += 1
            
            if step % args.log_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                accuracy = correct / total
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{accuracy:.4f}"
                })
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_accuracy = correct / total
        
        print(f"\nEpoch {epoch + 1} Training Results:")
        print(f"  Loss: {avg_epoch_loss:.4f}")
        print(f"  Accuracy: {train_accuracy:.4f}")
        
        if val_dataloader is not None and (epoch + 1) % args.eval_epochs == 0:
            print(f"\nRunning validation...")
            val_loss, val_accuracy = evaluate(model, val_dataloader, device)
            print(f"Validation Results:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  New best validation accuracy! Saving checkpoint...")
                checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")
                model.save_pretrained(checkpoint_dir)
                image_processor.save_pretrained(checkpoint_dir)
        
        if (epoch + 1) % args.save_epochs == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
            print(f"Saving checkpoint to {checkpoint_dir}...")
            model.save_pretrained(checkpoint_dir)
            image_processor.save_pretrained(checkpoint_dir)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    final_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model to {final_dir}...")
    model.save_pretrained(final_dir)
    image_processor.save_pretrained(final_dir)
    
    if val_dataloader is not None:
        print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

