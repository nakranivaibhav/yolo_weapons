import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from tqdm import tqdm

CV_FOLDS_PATH = Path("/workspace/yolo_dataset_cls_5fold")
OUTPUT_PATH = Path("/workspace/yolo_dataset_cls_5fold/predictions")
START_FOLD = 0
NUM_FOLDS = 1
EPOCHS = 1
MODEL_NAME = "facebook/convnextv2-base-22k-224"
BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_transforms(image_processor, is_train=True):
    if isinstance(image_processor.size, dict):
        size = image_processor.size.get("shortest_edge") or image_processor.size.get("height") or 224
    else:
        size = image_processor.size
    
    mean = image_processor.image_mean
    std = image_processor.image_std
    
    if is_train:
        transform = A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    return transform

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def train_fold(fold_idx, id2label, label2id):
    fold_dir = CV_FOLDS_PATH / f"fold_{fold_idx}"
    
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*60}")
    
    dataset = load_dataset("imagefolder", data_dir=str(fold_dir))
    print(f"Train samples: {len(dataset['train'])}")
    
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    train_transform = get_transforms(image_processor, is_train=True)
    
    def train_transforms(examples):
        examples["pixel_values"] = [
            train_transform(image=np.array(image.convert("RGB")))["image"] 
            for image in examples["image"]
        ]
        return examples
    
    train_dataset = dataset["train"].with_transform(train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model = ConvNextV2ForImageClassification.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Fold {fold_idx} Epoch {epoch + 1}/{EPOCHS}")
        
        for batch in progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = outputs.logits.argmax(-1)
            correct += (predicted == batch["labels"]).sum().item()
            total += batch["labels"].shape[0]
            
            progress_bar.set_postfix({
                "loss": f"{epoch_loss / (progress_bar.n + 1):.4f}",
                "acc": f"{correct / total:.4f}"
            })
    
    model_save_path = OUTPUT_PATH / f"fold_{fold_idx}_model"
    model.save_pretrained(model_save_path)
    image_processor.save_pretrained(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return model, image_processor, dataset

def collect_predictions(fold_idx, model, image_processor, dataset):
    print(f"\nCollecting predictions for fold {fold_idx}...")
    
    val_transform = get_transforms(image_processor, is_train=False)
    
    def val_transforms(examples):
        examples["pixel_values"] = [
            val_transform(image=np.array(image.convert("RGB")))["image"] 
            for image in examples["image"]
        ]
        return examples
    
    val_dataset = dataset["validation"].with_transform(val_transforms)
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Inference"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(pixel_values=batch["pixel_values"])
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    image_paths = [str(Path(dataset["validation"][i]["image"].filename)) for i in range(len(dataset["validation"]))]
    
    accuracy = (np.argmax(all_probs, axis=1) == all_labels).mean()
    print(f"Fold {fold_idx} Val Accuracy: {accuracy:.4f}")
    
    predictions = {
        'image_paths': image_paths,
        'pred_probs': all_probs,
        'labels': all_labels,
        'fold': fold_idx
    }
    
    pred_file = OUTPUT_PATH / f"fold_{fold_idx}_predictions.pkl"
    with open(pred_file, 'wb') as f:
        pickle.dump(predictions, f)
    
    return predictions

def merge_all_predictions(class_names):
    print(f"\n{'='*60}")
    print("MERGING ALL PREDICTIONS")
    print(f"{'='*60}")
    
    all_image_paths = []
    all_pred_probs = []
    all_labels = []
    all_folds = []
    
    for fold_idx in range(NUM_FOLDS):
        pred_file = OUTPUT_PATH / f"fold_{fold_idx}_predictions.pkl"
        with open(pred_file, 'rb') as f:
            fold_preds = pickle.load(f)
        
        all_image_paths.extend(fold_preds['image_paths'])
        all_pred_probs.append(fold_preds['pred_probs'])
        all_labels.append(fold_preds['labels'])
        all_folds.extend([fold_idx] * len(fold_preds['image_paths']))
    
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    merged = {
        'image_paths': all_image_paths,
        'pred_probs': all_pred_probs,
        'labels': all_labels,
        'folds': np.array(all_folds),
        'class_names': class_names
    }
    
    with open(OUTPUT_PATH / "all_predictions.pkl", 'wb') as f:
        pickle.dump(merged, f)
    
    overall_acc = (np.argmax(all_pred_probs, axis=1) == all_labels).mean()
    print(f"Total images: {len(all_image_paths)}")
    print(f"Overall CV Accuracy: {overall_acc:.4f}")
    
    return merged

def main():
    set_seed(SEED)
    
    print(f"\n{'#'*60}")
    print("5-FOLD CV TRAINING (CONVNEXT)")
    print(f"{'#'*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    first_fold_dataset = load_dataset("imagefolder", data_dir=str(CV_FOLDS_PATH / "fold_0"))
    class_names = first_fold_dataset["train"].features["label"].names
    id2label = {k: v for k, v in enumerate(class_names)}
    label2id = {v: k for k, v in enumerate(class_names)}
    
    print(f"Classes: {class_names}")
    
    with open(OUTPUT_PATH / "class_mapping.json", "w") as f:
        json.dump({"class_names": class_names}, f)
    
    for fold_idx in range(START_FOLD, NUM_FOLDS):
        model, image_processor, dataset = train_fold(fold_idx, id2label, label2id)
        collect_predictions(fold_idx, model, image_processor, dataset)
        del model
        torch.cuda.empty_cache()
    
    merge_all_predictions(class_names)
    
    print(f"\nDone! Predictions saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()