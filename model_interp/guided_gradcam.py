"""
GuidedGradCam for YOLO Weapon Detection Model
Combines GradCAM localization with Guided Backpropagation for sharper saliency maps.
"""
import os
import sys
import glob
import argparse
from tqdm import tqdm

import numpy as np
import cv2

import torch
import torch.nn as nn
from captum.attr import GuidedGradCam, LayerGradCam
from ultralytics.data.augment import LetterBox


# ---------------- YOLO Wrapper ----------------
class YOLOWrapper(nn.Module):
    """
    Wrapper for YOLO model that returns a scalar for the top detection.
    Works with weapon detection models (classes: knife, gun, rifle, baseball_bat).
    """
    def __init__(self, model, target_class=None):
        super().__init__()
        self.model = model
        self.target_class = target_class  # None = use highest conf, or specify 0-3
        self.last_score = 0.0
        self.last_class_id = 0
        self.last_box = None

    def forward(self, input_tensor):
        preds = self.model(input_tensor)
        
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        if isinstance(preds, dict) and 'pred_logits' in preds:
            logits = preds['pred_logits'][0]
            probs = logits.softmax(dim=-1)
        elif isinstance(preds, torch.Tensor) and preds.dim() == 3:
            # YOLO format: [B, N, 4+num_classes] or [B, N, 5+num_classes]
            if preds.shape[-1] > 5:
                confs = preds[0, :, 4]
                class_logits = preds[0, :, 5:]
                probs = torch.softmax(class_logits, dim=-1)
                combined = confs.unsqueeze(-1) * probs
            else:
                # Simple format
                confs = preds[0, :, 4]
                combined = confs.unsqueeze(-1)
                probs = combined
        else:
            return (input_tensor * 0).sum().view(1)
        
        if self.target_class is not None:
            # Find best detection for target class
            class_scores = combined[:, self.target_class] if combined.dim() > 1 else combined.squeeze()
            best_idx = class_scores.argmax()
            score = class_scores[best_idx]
            self.last_class_id = self.target_class
        else:
            # Find overall best detection
            if combined.dim() > 1:
                best_idx = combined.max(dim=-1).values.argmax()
                best_class = combined[best_idx].argmax()
                score = combined[best_idx, best_class]
                self.last_class_id = int(best_class.item())
            else:
                best_idx = combined.argmax()
                score = combined[best_idx]
                self.last_class_id = 0
        
        self.last_score = float(score.detach().cpu().item())
        
        # Store bounding box if available
        if preds.shape[-1] >= 4:
            self.last_box = preds[0, best_idx, :4].detach().cpu().numpy()
        
        return score.unsqueeze(0)


# ---------------- Preprocessing ----------------
def preprocess_image(img_path, img_size=640, device='cpu'):
    """Load image, apply LetterBox, return tensor and original image."""
    img0 = cv2.imread(img_path)
    if img0 is None:
        return None, None, None
    
    lb = LetterBox(img_size, auto=False, stride=32)
    img = lb(image=img0)
    
    img_t = img.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).float().to(device) / 255.0
    
    if img_t.dim() == 3:
        img_t = img_t.unsqueeze(0)
    
    return img_t, img0, img


# ---------------- Layer Selection ----------------
def find_last_conv(module):
    """Return the last Conv2d layer in the model."""
    last = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = (name, m)
    return last


# ---------------- Visualization ----------------
def smart_resize_heatmap(heatmap_raw, original_shape, model_size=(640, 640)):
    """Reverse LetterBox padding and resize to original image shape."""
    orig_h, orig_w = original_shape[:2]
    mh, mw = model_size
    scale = min(mh / orig_h, mw / orig_w)
    nw, nh = int(orig_w * scale), int(orig_h * scale)
    dw, dh = (mw - nw) // 2, (mh - nh) // 2
    heatmap_cropped = heatmap_raw[dh:dh+nh, dw:dw+nw]
    return cv2.resize(heatmap_cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def create_overlay(original_image, heatmap, alpha=0.5):
    """Create heatmap overlay on original image."""
    h, w = original_image.shape[:2]
    
    # Resize heatmap to match original
    if heatmap.shape[:2] != (h, w):
        heatmap = smart_resize_heatmap(heatmap, (h, w))
    
    # Normalize
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ---------------- Model Loading ----------------
def load_model(weights_path, device):
    """Load YOLO model from weights file."""
    try:
        from ultralytics import YOLO
        y = YOLO(weights_path)
        model = y.model.float().to(device).eval()
        return model
    except Exception as e:
        print(f"Failed to load with ultralytics: {e}")
        ckpt = torch.load(weights_path, weights_only=False, map_location=device)
        model = ckpt['model'] if 'model' in ckpt else ckpt
        return model.float().to(device).eval()


# ---------------- GuidedGradCam Attribution ----------------
def compute_guided_gradcam(wrapped_model, target_layer, input_tensor, model_size=640):
    """Compute GuidedGradCam attribution."""
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    ggc = GuidedGradCam(wrapped_model, target_layer)

    try:
        attributions = ggc.attribute(input_tensor, target=None)
    except Exception as e:
        print(f"GuidedGradCam failed: {e}, falling back to LayerGradCam")
        lgc = LayerGradCam(wrapped_model, target_layer)
        attributions = lgc.attribute(input_tensor)
        attributions = LayerGradCam.interpolate(attributions, (model_size, model_size))

    # Process attribution to heatmap
    attr_np = attributions.detach().cpu().numpy()[0]

    # Sum across channels if needed
    if attr_np.ndim == 3:
        heatmap = np.sum(np.abs(attr_np), axis=0)
    else:
        heatmap = np.abs(attr_np)

    # Resize to model size if needed
    if heatmap.shape[0] != model_size:
        heatmap = cv2.resize(heatmap, (model_size, model_size), interpolation=cv2.INTER_CUBIC)

    return heatmap


WEAPON_LABELS = {0: 'knife', 1: 'gun', 2: 'rifle', 3: 'baseball_bat'}


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='GuidedGradCam for YOLO Weapon Detection')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Input: image file, folder, or video')
    parser.add_argument('--out', type=str, required=True, help='Output folder or video path')
    parser.add_argument('--model-size', type=int, default=640, help='Model input size')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Target class ID (0=knife, 1=gun, 2=rifle, 3=bat). None=best detection')
    parser.add_argument('--conf-thresh', type=float, default=0.2, help='Min confidence threshold')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay alpha (0-1)')
    parser.add_argument('--layer-name', type=str, default=None, help='Specific layer name (auto if None)')
    parser.add_argument('--limit', type=int, default=None, help='Limit frames/images to process')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.weights}")
    model = load_model(args.weights, device)
    wrapped = YOLOWrapper(model, target_class=args.target_class)

    # Find target layer
    target_layer = None
    if args.layer_name:
        for name, mod in model.named_modules():
            if name == args.layer_name:
                target_layer = mod
                print(f"Using specified layer: {name}")
                break

    if target_layer is None:
        conv_info = find_last_conv(model)
        if conv_info:
            target_layer = conv_info[1]
            print(f"Auto-selected layer: {conv_info[0]}")
        else:
            raise RuntimeError("No Conv2d layer found in model")

    # Determine input type
    source_path = args.source
    is_video = source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        process_video(args, wrapped, target_layer, device)
    else:
        process_images(args, wrapped, target_layer, device)


def process_images(args, wrapped, target_layer, device):
    """Process image files or folder."""
    source = args.source

    if os.path.isdir(source):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(source, ext)))
        files = sorted(files)
    else:
        files = [source]

    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    print(f"Processing {len(files)} images...")

    for img_path in tqdm(files):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.out, f"ggcam_{fname}")

        img_t, img0, img_resized = preprocess_image(img_path, args.model_size, device)
        if img_t is None:
            continue

        # Get detection score
        with torch.no_grad():
            _ = wrapped(img_t)
            score = wrapped.last_score
            cls_id = wrapped.last_class_id

        if score < args.conf_thresh:
            # No detection - save grayed image
            gray = (img0 * 0.3).astype(np.uint8)
            cv2.putText(gray, "NO DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(out_path, gray)
            continue

        # Compute GuidedGradCam
        try:
            heatmap = compute_guided_gradcam(wrapped, target_layer, img_t, args.model_size)
            overlay = create_overlay(img0, heatmap, args.alpha)

            # Add label
            label = WEAPON_LABELS.get(cls_id, str(cls_id))
            cv2.putText(overlay, f"{label}: {score:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imwrite(out_path, overlay)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"Done! Results saved to {args.out}")


def process_video(args, wrapped, target_layer, device):
    """Process video file."""
    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    limit = args.limit if args.limit else total_frames
    print(f"Processing video: {width}x{height} @ {fps}FPS ({total_frames} frames, limit={limit})")

    # Output at model size for alignment
    out_size = (args.model_size, args.model_size)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

    frame_count = 0
    pbar = tqdm(total=limit, desc="Processing")

    while cap.isOpened() and frame_count < limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        lb = LetterBox(args.model_size, auto=False, stride=32)
        img_resized = lb(image=frame)
        img_t = img_resized.transpose((2, 0, 1))[::-1]
        img_t = np.ascontiguousarray(img_t)
        img_t = torch.from_numpy(img_t).float().to(device) / 255.0
        img_t = img_t.unsqueeze(0)

        try:
            # Get detection
            with torch.no_grad():
                _ = wrapped(img_t)
                score = wrapped.last_score
                cls_id = wrapped.last_class_id

            if score >= args.conf_thresh:
                img_t_grad = img_t.clone().detach().requires_grad_(True)
                heatmap = compute_guided_gradcam(wrapped, target_layer, img_t_grad, args.model_size)
                overlay = create_overlay(img_resized, heatmap, args.alpha)

                label = WEAPON_LABELS.get(cls_id, str(cls_id))
                cv2.putText(overlay, f"{label}: {score:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                overlay = img_resized.copy()
                cv2.putText(overlay, "No detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

            writer.write(overlay)
        except Exception as e:
            print(f"\nError on frame {frame_count}: {e}")
            writer.write(img_resized)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"Done! Saved to {args.out}")


if __name__ == "__main__":
    main()

