#!/usr/bin/env python3
"""
yolo11_ig_images.py
Run YOLOv11 + Integrated Gradients on a folder of images and save localized IG overlays.

Usage:
    python yolo11_ig_images.py --source ./images --weights yolo11_best.pt --out ./out --model_size 640
"""
import argparse
from pathlib import Path
import json
import time
import sys

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel
from scipy.ndimage import binary_dilation

# Try to import ultralytics components
try:
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
except Exception as e:
    raise RuntimeError("Install ultralytics (YOLOv11) and ensure it's importable.") from e

# ---------------- Utilities ----------------
def safe_float(x, default=0.0):
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        return default

def smart_resize_heatmap(heatmap_raw, original_shape, model_size=(640,640)):
    """
    Reverse LetterBox: crop valid region from model-canvas heatmap and resize to original image shape.
    heatmap_raw: Hc x Wc (model_size x model_size by convention)
    original_shape: (H_orig, W_orig, ...)
    """
    orig_h, orig_w = original_shape[:2]
    mh, mw = model_size
    scale = min(mh / orig_h, mw / orig_w)
    nw = int(orig_w * scale)
    nh = int(orig_h * scale)
    dw = (mw - nw) // 2
    dh = (mh - nh) // 2
    # Guard bounds
    h, w = heatmap_raw.shape[:2]
    # Crop to the centered valid region
    crop = heatmap_raw[dh:dh+nh, dw:dw+nw]
    if crop.size == 0:
        # fallback: resize entire heatmap
        return cv2.resize(heatmap_raw, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

def clip_box(x1, y1, x2, y2, W, H):
    x1c = max(0, min(W-1, int(round(x1))))
    y1c = max(0, min(H-1, int(round(y1))))
    x2c = max(0, min(W, int(round(x2))))
    y2c = max(0, min(H, int(round(y2))))
    if x2c <= x1c or y2c <= y1c:
        return None
    return (x1c, y1c, x2c, y2c)

# ---------------- YOLO wrapper for tensor-forward attribution ----------------
class YOLOWrapper(nn.Module):
    """
    Wrap underlying YOLO nn.Module (expects forward to return [B, N, 5+K]).
    Returns a scalar for the selected detection (combined conf*class_prob).
    Set wrapper._forced_index to force attribution for that detection index.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_score = 0.0
        self.last_class_id = 0
        self._forced_index = None

    def forward(self, input_tensor):
        preds = self.model(input_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, torch.Tensor) and preds.dim() == 3:
            # preds [B, N, 5+K]
            confs = preds[0, :, 4]
            class_logits = preds[0, :, 5:]
            if class_logits.numel() == 0:
                combined = confs
                cls_ids = torch.zeros_like(confs, dtype=torch.long)
            else:
                probs = torch.softmax(class_logits, dim=-1)
                cls_ids = torch.argmax(probs, dim=-1)
                combined = confs * probs.max(dim=-1).values
            if combined.numel() == 0:
                return (input_tensor * 0).sum().view(1)
            idx = int(self._forced_index) if self._forced_index is not None else int(torch.argmax(combined))
            s = float(combined[idx].detach().cpu().item())
            c = int(cls_ids[idx].detach().cpu().item())
            self.last_score = s
            self.last_class_id = c
            return combined[idx].unsqueeze(0)
        # fallback
        return (input_tensor * 0).sum().view(1)

# ---------------- Integrated Gradients (positive-only) ----------------
def ig_for_detection_positive(wrapped, input_tensor, det_index, model_size=640, device='cpu', nt_samples=6, internal_batch_size=1):
    """
    Force wrapper to attribute to detection det_index and return model_size x model_size heatmap 0..1.
    """
    wrapped._forced_index = int(det_index)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    # blurred baseline
    img_np = input_tensor.detach().cpu().numpy()[0].transpose(1,2,0)
    img_np = np.clip(img_np, 0.0, 1.0)
    img_blur = cv2.GaussianBlur((img_np * 255).astype(np.uint8), (51,51), 0).astype(np.float32) / 255.0
    baseline = torch.from_numpy(img_blur.transpose(2,0,1)).unsqueeze(0).float().to(device)

    ig = IntegratedGradients(wrapped)
    nt = NoiseTunnel(ig)

    attributions = nt.attribute(
        input_tensor,
        baselines=baseline,
        nt_type='smoothgrad_sq',
        nt_samples=max(3, nt_samples),
        stdevs=0.08,
        internal_batch_size=internal_batch_size
    )
    attr = attributions.detach().cpu().numpy()[0]  # C,H,W

    pos_attr = np.clip(attr, 0, None)
    heat = np.sum(pos_attr, axis=0)  # Hf x Wf
    if heat.max() <= 0:
        wrapped._forced_index = None
        return np.zeros((model_size, model_size), dtype=np.float32)

    heat_up = cv2.resize(heat, (model_size, model_size), interpolation=cv2.INTER_CUBIC)
    p99 = np.percentile(heat_up, 99)
    if p99 > 0:
        heat_up = np.clip(heat_up, 0, p99) / (p99 + 1e-9)
    wrapped._forced_index = None
    return heat_up

# ---------------- Localized overlay tuned for the look you provided ----------------
def overlay_localized_heatmap(original_image, heatmap_model, bbox_model, model_size=640,
                              box_expand=0.08, blur_ks=(21,21), mask_dilate_iter=3,
                              outside_attenuation=0.12, overlay_alpha=0.6):
    """
    original_image: HxW BGR (full-res)
    heatmap_model: model_size x model_size (float 0..1)
    bbox_model: (x1,y1,x2,y2) coordinates in model-canvas (0..model_size)
    Returns: composed_image, bbox_full (in full-res)
    """
    H, W = original_image.shape[:2]
    heat_aligned = smart_resize_heatmap(heatmap_model, (H, W), model_size=(model_size, model_size))
    heat_aligned = np.clip(heat_aligned, 0.0, 1.0)

    # map bbox model->full-res via reverse-LetterBox math
    mh, mw = model_size, model_size
    scale = min(mh / H, mw / W)
    nw = int(W * scale); nh = int(H * scale)
    dw = (mw - nw) // 2; dh = (mh - nh) // 2

    x1_m, y1_m, x2_m, y2_m = bbox_model
    bx1 = max(0, x1_m - dw); by1 = max(0, y1_m - dh)
    bx2 = max(0, x2_m - dw); by2 = max(0, y2_m - dh)

    bx1_full = int(round(bx1 / scale)); by1_full = int(round(by1 / scale))
    bx2_full = int(round(bx2 / scale)); by2_full = int(round(by2 / scale))

    # expand a little
    w_box = max(1, bx2_full - bx1_full); h_box = max(1, by2_full - by1_full)
    ex_w = int(round(w_box * box_expand)); ex_h = int(round(h_box * box_expand))
    bx1_full = max(0, bx1_full - ex_w); by1_full = max(0, by1_full - ex_h)
    bx2_full = min(W, bx2_full + ex_w); by2_full = min(H, by2_full + ex_h)

    # mask & dilate
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[by1_full:by2_full, bx1_full:bx2_full] = 1
    if mask_dilate_iter > 0:
        mask = binary_dilation(mask, iterations=mask_dilate_iter).astype(np.uint8)

    # local normalization inside mask
    local_vals = heat_aligned[mask == 1]
    heat_vis = heat_aligned.copy()
    if local_vals.size == 0 or np.nanmax(local_vals) <= 0:
        if heat_aligned.max() > 0:
            heat_vis = heat_aligned / (np.percentile(heat_aligned, 99) + 1e-9)
    else:
        local_p99 = np.percentile(local_vals, 99)
        local_p1 = np.percentile(local_vals, 1)
        if local_p99 - local_p1 > 1e-6:
            heat_vis[mask == 1] = np.clip((heat_vis[mask == 1] - local_p1) / (local_p99 - local_p1), 0, 1)
        else:
            heat_vis[mask == 1] = np.clip(heat_vis[mask == 1] / (local_p99 + 1e-9), 0, 1)
        heat_vis[mask == 0] *= outside_attenuation

    # smooth & normalize a bit
    heat_vis = cv2.GaussianBlur(heat_vis, blur_ks, 0)
    if heat_vis.max() > 0:
        heat_vis = heat_vis / (np.percentile(heat_vis, 99) + 1e-9)

    # colorize
    heat_uint8 = np.uint8(255 * np.clip(heat_vis, 0, 1))
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # dark background
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray_dark = (gray_bgr * 0.28).astype(np.uint8)

    # alpha by intensity
    alpha_map = (heat_uint8.astype(np.float32) / 255.0) ** 0.9
    alpha_map = cv2.merge([alpha_map, alpha_map, alpha_map])
    blended = (heat_color.astype(np.float32) * alpha_map + gray_dark.astype(np.float32) * (1 - alpha_map)).astype(np.uint8)

    composed = cv2.addWeighted(blended, overlay_alpha, original_image, 1 - overlay_alpha, 0)
    return composed, (bx1_full, by1_full, bx2_full, by2_full)

# ---------------- Preprocessing with LetterBox ----------------
def preprocess_image_with_letterbox(img_bgr, model_size=640, device='cpu'):
    """
    Use Ultralytics LetterBox to reproduce training preprocessing.
    Returns tensor [1,C,Hf,Wf] float 0..1 and the padded/resized image (HxW).
    """
    lb = LetterBox(model_size, auto=False, stride=32)
    out = lb(image=img_bgr)
    if isinstance(out, dict) and 'image' in out:
        img_out = out['image']
    else:
        img_out = out
    img_t = img_out.transpose((2,0,1))[::-1]  # CHW and reverse channels as earlier scripts
    img_t = np.ascontiguousarray(img_t)
    tensor = torch.from_numpy(img_t).float().to(device) / 255.0
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor, img_out

# ---------------- Main script ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Folder with input images')
    p.add_argument('--weights', required=True, help='YOLOv11 weights (.pt)')
    p.add_argument('--out', required=True, help='Output folder for overlays')
    p.add_argument('--model_size', type=int, default=640)
    p.add_argument('--nt_samples', type=int, default=8)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_images', type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print("Device:", device)

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load YOLOv11 high-level and underlying model
    print("Loading YOLOv11 weights:", args.weights)
    y = YOLO(args.weights)
    y_model = y.model.to(device).eval()
    print("  ✓ YOLO loaded")

    # wrap underlying model for attribution (raw forward)
    wrapper = YOLOWrapper(y_model)

    # list images
    exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.webp']
    files = []
    for e in exts:
        files.extend(list(src.glob(e)))
    files = sorted(files)
    if args.max_images:
        files = files[:args.max_images]
    print(f"Found {len(files)} images")

    stats = {'imgs':0, 'total_dets':0}
    t0 = time.perf_counter()

    for img_path in tqdm(files, desc="Images"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("WARN: couldn't read", img_path)
            continue
        stats['imgs'] += 1

        H, W = img_bgr.shape[:2]

        # prepare model input with LetterBox (ensures coordinate conventions match)
        input_t, model_canvas_img = preprocess_image_with_letterbox(img_bgr, model_size=args.model_size, device=device)
        # use high-level y(...) to get detections in model canvas coords
        # Note: YOLO high-level accepts images or tensors; to ensure consistent coords use model_canvas_img passed as numpy
        dets = y(model_canvas_img, conf=args.conf, verbose=False)[0]  # Detections object
        # parse boxes robustly
        boxes_xyxy = []
        if hasattr(dets, 'boxes'):
            try:
                # dets.boxes.xyxy likely tensor Nx4
                boxes = dets.boxes.xyxy.cpu().numpy() if hasattr(dets.boxes.xyxy, 'cpu') else np.array(dets.boxes.xyxy)
                confs = dets.boxes.conf.cpu().numpy() if hasattr(dets.boxes.conf, 'cpu') else np.array(dets.boxes.conf)
                cls_ids = dets.boxes.cls.cpu().numpy() if hasattr(dets.boxes.cls, 'cpu') else np.array(dets.boxes.cls)
                for b, cf, cl in zip(boxes, confs, cls_ids):
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    boxes_xyxy.append((x1, y1, x2, y2, float(cf), int(cl)))
            except Exception:
                # fallback parse
                try:
                    for box in dets.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        cf = float(box.conf[0])
                        cl = int(box.cls[0]) if hasattr(box, 'cls') else 0
                        boxes_xyxy.append((x1, y1, x2, y2, cf, cl))
                except Exception:
                    boxes_xyxy = []

        # if no detections, just save a dimmed image to indicate none
        if len(boxes_xyxy) == 0:
            out_img = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
            out_img = cv2.cvtColor((out_img * 0.4).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            out_path = out_dir / f"{img_path.stem}_no_det{img_path.suffix}"
            cv2.imencode(img_path.suffix, out_img)[1].tofile(str(out_path))
            continue

        stats['total_dets'] += len(boxes_xyxy)

        # For each detection, compute per-detection IG and overlay localized heatmap, blending results.
        composed_canvas = img_bgr.copy()
        # We'll alpha-blend each detection overlay sequentially to the composed_canvas
        for det_idx, (x1_m, y1_m, x2_m, y2_m, conf_m, cls_m) in enumerate(boxes_xyxy):
            # For IG, we need the *same input tensor* that corresponds to this model canvas
            # input_t is result of LetterBox(image=img_bgr), so forward the underlying model on input_t to ensure same indexing
            # Compute per-detection IG
            heat_model = ig_for_detection_positive(wrapper, input_t, det_index=det_idx,
                                                  model_size=args.model_size, device=device,
                                                  nt_samples=args.nt_samples, internal_batch_size=1)
            # overlay localized heatmap onto original full-res image
            composed_canvas, bbox_full = overlay_localized_heatmap(composed_canvas, heat_model,
                                                                   (x1_m, y1_m, x2_m, y2_m),
                                                                   model_size=args.model_size,
                                                                   box_expand=0.08, blur_ks=(21,21),
                                                                   mask_dilate_iter=3, outside_attenuation=0.12,
                                                                   overlay_alpha=0.6)
            # draw a tight rectangle (optional)
            bx1, by1, bx2, by2 = bbox_full
            color = (0,0,255) if int(cls_m) in (1,2) else (0,255,255)
            cv2.rectangle(composed_canvas, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(composed_canvas, f"{int(cls_m)} {conf_m:.2f}", (bx1, max(12, by1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # save result
        out_path = out_dir / f"{img_path.stem}_ig{img_path.suffix}"
        cv2.imencode(img_path.suffix, composed_canvas)[1].tofile(str(out_path))

    elapsed = time.perf_counter() - t0
    print(f"Done. Processed {stats['imgs']} images, total_detections={stats['total_dets']}, time={elapsed:.1f}s")

if __name__ == "__main__":
    main()
