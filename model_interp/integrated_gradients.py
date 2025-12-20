#!/usr/bin/env python3
"""
YOLOv11 + Integrated Gradients (IMAGE FOLDER)
Low-VRAM, autograd-safe, stable implementation.

- NO NoiseTunnel
- Positive-only IG
- Local bbox normalization
- YOLOv11 only
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy.ndimage import binary_dilation

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

# -------------------- Utils --------------------

def smart_resize_heatmap(heat, orig_shape, model_size):
    H, W = orig_shape[:2]
    scale = min(model_size / H, model_size / W)
    nh, nw = int(H * scale), int(W * scale)
    dh, dw = (model_size - nh) // 2, (model_size - nw) // 2
    crop = heat[dh:dh+nh, dw:dw+nw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)

# -------------------- YOLO Wrapper --------------------

class YOLOWrapper(nn.Module):
    """
    Wrap raw YOLO nn.Module.
    Returns a scalar score for ONE detection index.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.target_index = None

    def forward(self, x):
        preds = self.model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # preds: [B, N, 5+K]
        conf = preds[0, :, 4]
        cls_logits = preds[0, :, 5:]
        probs = torch.softmax(cls_logits, dim=-1)
        scores = conf * probs.max(dim=-1).values

        if scores.numel() == 0:
            return (x * 0).sum().view(1)

        idx = self.target_index if self.target_index is not None else torch.argmax(scores)
        return scores[idx].unsqueeze(0)

# -------------------- Integrated Gradients --------------------

def compute_ig_heatmap(wrapper, input_tensor, det_idx, model_size):
    """
    Compute positive-only Integrated Gradients heatmap for ONE detection.
    """
    wrapper.target_index = int(det_idx)

    # blurred baseline
    img = input_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    blur = cv2.GaussianBlur((img * 255).astype(np.uint8), (51, 51), 0) / 255.0
    baseline = torch.from_numpy(blur.transpose(2, 0, 1)).unsqueeze(0).float().to(input_tensor.device)

    ig = IntegratedGradients(wrapper)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=16,
        internal_batch_size=1
    )

    attr = attributions.detach().cpu().numpy()[0]

    # positive-only
    attr = np.clip(attr, 0, None)
    heat = np.sum(attr, axis=0)

    # remove top-left artifact
    heat[:6, :6] = 0

    heat = cv2.resize(heat, (model_size, model_size), interpolation=cv2.INTER_CUBIC)
    p99 = np.percentile(heat, 99)
    if p99 > 0:
        heat = np.clip(heat, 0, p99) / (p99 + 1e-9)

    wrapper.target_index = None
    return heat

# -------------------- Visualization --------------------

def overlay_heatmap(image, heat, bbox, model_size):
    H, W = image.shape[:2]
    heat = smart_resize_heatmap(heat, image.shape, model_size)

    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(0.08 * bw))
    y1 = max(0, y1 - int(0.08 * bh))
    x2 = min(W, x2 + int(0.08 * bw))
    y2 = min(H, y2 + int(0.08 * bh))

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    mask = binary_dilation(mask, iterations=3)

    local = heat[mask == 1]
    if local.size > 0:
        p1, p99 = np.percentile(local, 1), np.percentile(local, 99)
        heat[mask == 1] = np.clip((heat[mask == 1] - p1) / (p99 - p1 + 1e-9), 0, 1)
    heat[mask == 0] *= 0.12

    heat = cv2.GaussianBlur(heat, (31, 31), 0)

    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = (gray * 0.28).astype(np.uint8)

    alpha = (heat_u8 / 255.0) ** 0.9
    alpha = cv2.merge([alpha, alpha, alpha])

    overlay = (heat_color * alpha + gray * (1 - alpha)).astype(np.uint8)
    out = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_size", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_dets", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    wrapper = YOLOWrapper(model)

    lb = LetterBox(args.model_size, auto=False, stride=32)

    images = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp"):
        images += list(src.glob(ext))

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        inp = lb(image=img)
        if isinstance(inp, dict):
            inp = inp["image"]

        t = inp[..., ::-1].transpose(2, 0, 1).copy()
        t = torch.from_numpy(t).float().unsqueeze(0).to(device) / 255.0

        # Detection (NO gradients)
        with torch.no_grad():
            dets = yolo(inp, conf=args.conf, verbose=False)[0]

        torch.cuda.empty_cache()

        if not hasattr(dets, "boxes") or len(dets.boxes) == 0:
            continue

        canvas = img.copy()

        for i, box in enumerate(dets.boxes[:args.max_dets]):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # clean tensor for IG
            t_ig = t.clone().detach().requires_grad_(True)

            with torch.inference_mode(False):
                with torch.enable_grad():
                    heat = compute_ig_heatmap(
                        wrapper, t_ig, i, args.model_size
                    )

            canvas = overlay_heatmap(canvas, heat, (x1, y1, x2, y2), args.model_size)

            del t_ig
            torch.cuda.empty_cache()

        cv2.imwrite(str(out_dir / f"{img_path.stem}_ig{img_path.suffix}"), canvas)

    print("✅ Done.")

if __name__ == "__main__":
    main()
