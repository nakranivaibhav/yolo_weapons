#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
from datasets import load_dataset
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXT model on test set")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "train"], help="Dataset split to evaluate")
    return parser.parse_args()


def load_model_and_processor(model_dir, device):
    print(f"Loading model from {model_dir}...")
    model = ConvNextV2ForImageClassification.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, image_processor


def get_eval_transforms(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = image_processor.size["shortest_edge"]
    
    transform = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize
    ])
    
    return transform


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate_model(model, dataloader, device):
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_labels = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(pixel_values=batch["pixel_values"])
            logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = logits.argmax(-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)
            all_probabilities.extend(probabilities)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def compute_metrics(predictions, labels, label_names):
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    cm = confusion_matrix(labels, predictions)
    
    report = classification_report(labels, predictions, target_names=label_names, digits=4)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class_metrics": {
            label_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i])
            }
            for i in range(len(label_names))
        },
        "confusion_matrix": cm.tolist()
    }
    
    return metrics, cm, report


def plot_confusion_matrix(cm, label_names, output_path):
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()


def plot_normalized_confusion_matrix(cm, label_names, output_path):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to: {output_path}")
    plt.close()


def print_metrics_summary(metrics, label_names):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\nMacro Average (unweighted):")
    print(f"  Precision:          {metrics['precision_macro']:.4f}")
    print(f"  Recall:             {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:           {metrics['f1_macro']:.4f}")
    print(f"\nWeighted Average:")
    print(f"  Precision:          {metrics['precision_weighted']:.4f}")
    print(f"  Recall:             {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:           {metrics['f1_weighted']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for class_name in label_names:
        class_metrics = metrics['per_class_metrics'][class_name]
        print(f"\n  {class_name.upper()}:")
        print(f"    Precision:        {class_metrics['precision']:.4f}")
        print(f"    Recall:           {class_metrics['recall']:.4f}")
        print(f"    F1-Score:         {class_metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  {'':>10} | " + " | ".join([f"{name:>10}" for name in label_names]))
    print(f"  {'-'*12}|" + "|".join(["-"*12 for _ in label_names]))
    for i, name in enumerate(label_names):
        print(f"  {name:>10} | " + " | ".join([f"{cm[i][j]:>10}" for j in range(len(label_names))]))
    
    print("\n" + "="*60)


def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, image_processor = load_model_and_processor(args.model_dir, device)
    
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    
    split_mapping = {
        "validation": "validation" if "validation" in dataset else "valid",
        "test": "test",
        "train": "train"
    }
    
    eval_split = split_mapping.get(args.split, args.split)
    if eval_split not in dataset:
        if args.split == "validation" and "valid" in dataset:
            eval_split = "valid"
        else:
            raise ValueError(f"Split '{args.split}' not found in dataset. Available splits: {list(dataset.keys())}")
    
    label_names = dataset["train"].features["label"].names
    print(f"Classes: {label_names}")
    print(f"Evaluating on '{eval_split}' split with {len(dataset[eval_split])} samples")
    
    eval_transform = get_eval_transforms(image_processor)
    
    def transform_fn(examples):
        examples["pixel_values"] = [eval_transform(image.convert("RGB")) for image in examples["image"]]
        return examples
    
    eval_dataset = dataset[eval_split].with_transform(transform_fn)
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    predictions, labels, probabilities = evaluate_model(model, eval_dataloader, device)
    
    metrics, cm, report = compute_metrics(predictions, labels, label_names)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    report_file = os.path.join(args.output_dir, f"classification_report_{args.split}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_file}")
    
    cm_plot = os.path.join(args.output_dir, f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(cm, label_names, cm_plot)
    
    cm_norm_plot = os.path.join(args.output_dir, f"confusion_matrix_normalized_{args.split}.png")
    plot_normalized_confusion_matrix(cm, label_names, cm_norm_plot)
    
    print_metrics_summary(metrics, label_names)
    
    print(f"\n\nDetailed Classification Report:")
    print(report)
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

