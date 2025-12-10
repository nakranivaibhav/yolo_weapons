# deyo_gradcam.py
# Use: python deyo_gradcam.py --weights /path/to/model.pt --source ./images --out ./out --model-size 640

import torch
import cv2
import numpy as np
import argparse
import sys
import os
import glob
from tqdm import tqdm
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients, NoiseTunnel
from ultralytics.data.augment import LetterBox
import math

# ----------------- Helpers (same semantics as yours) -----------------
def safe_int(value, default=0):
    try:
        return int(value.item()) if isinstance(value, torch.Tensor) else int(value)
    except:
        return default

def safe_float(value, default=0.0):
    try:
        return float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    except:
        return default

def smart_resize_heatmap(heatmap_raw, original_shape, model_size=(640, 640)):
    orig_h, orig_w = original_shape[:2]
    mh, mw = model_size
    scale = min(mh / orig_h, mw / orig_w)
    nw = int(orig_w * scale)
    nh = int(orig_h * scale)
    dw = (mw - nw) // 2
    dh = (mh - nh) // 2
    heatmap_cropped = heatmap_raw[dh:dh+nh, dw:dw+nw]
    heatmap_final = cv2.resize(heatmap_cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return heatmap_final

def create_thermal_overlay(original_image, heatmap_raw):
    heatmap_raw[:6, :6] = 0
    h, w = original_image.shape[:2]
    heatmap_aligned = smart_resize_heatmap(heatmap_raw, (h, w))
    heatmap_blurred = cv2.GaussianBlur(heatmap_aligned, (31, 31), 0)
    if np.max(heatmap_blurred) > 0:
        v_max = np.percentile(heatmap_blurred, 99)
        heatmap_blurred = np.clip(heatmap_blurred, 0, v_max)
        norm = heatmap_blurred / (v_max + 1e-8)
        heatmap_uint8 = np.uint8(255 * norm)
    else:
        heatmap_uint8 = heatmap_blurred.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray_dark = (gray_bgr * 0.4).astype(np.uint8)
    mask = heatmap_uint8 > 40
    final_image = gray_dark.copy()
    final_image[mask] = cv2.addWeighted(gray_dark[mask], 0.3, heatmap_color[mask], 0.7, 0)
    return final_image

# ----------------- Preprocess -----------------
def preprocess_image(img_path, img_size=640, device='cpu'):
    img0 = cv2.imread(img_path)
    if img0 is None: return None, None
    lb = LetterBox(img_size, auto=False, stride=32)
    img = lb(image=img0)
    if isinstance(img, dict) and 'image' in img:
        img = img['image']
    # original script reversed channels; keep that behaviour
    img_t = img.transpose((2, 0, 1))[::-1]
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).float().to(device)
    img_t /= 255.0
    if len(img_t.shape) == 3: img_t = img_t[None]
    return img_t, img0

# ----------------- Model wrapper that returns scalar per selected detection -----------------
class DEYOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_score = 0.0
        self.last_class_id = 0
        self._forced_target = None

    def forward(self, input_tensor):
        preds = self.model(input_tensor)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        if isinstance(preds, dict) and 'pred_logits' in preds:
            logits = preds['pred_logits'][0]  # [Q, C+1]
        else:
            try:
                logits = preds[0, :, 4:]
            except Exception:
                flattened = preds.flatten()
                idx = torch.argmax(flattened)
                self.last_score = float(flattened[idx].detach().abs().cpu().item())
                return flattened[idx].unsqueeze(0)
        # compute probs for selection
        try:
            probs = logits.softmax(-1)
        except Exception:
            probs = logits.sigmoid()
        max_scores, class_ids = probs.max(dim=-1)
        best_query_idx = max_scores.argmax()
        target_class_idx = safe_int(class_ids[best_query_idx])
        self.last_score = max_scores[best_query_idx].detach().cpu().item()
        self.last_class_id = int(target_class_idx)
        if safe_float(self.last_score) < 0.1:
            return (input_tensor * 0).sum().view(1)
        if isinstance(self._forced_target, tuple) and len(self._forced_target) == 2:
            qidx, cidx = self._forced_target
            return logits[int(qidx), int(cidx)].unsqueeze(0)
        return logits[best_query_idx, target_class_idx].unsqueeze(0)

# ----------------- Helper: find target conv layer -----------------
def find_last_conv(model):
    # prefer backbone convs: search reverse order for nn.Conv2d
    last_conv = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            last_conv = (name, module)
            break
    return last_conv

# ----------------- Grad-CAM attribution function -----------------
def gradcam_for_image(wrapped_model, raw_model, input_tensor, target_layer, device, model_size=640):
    """
    wrapped_model: DEYOWrapper (returns scalar per detection)
    raw_model: the original nn.Module used to select a layer
    target_layer: module object (nn.Conv2d)
    input_tensor: [1,3,H,W] normalized
    returns: heatmap upsampled to model_size x model_size (numpy float 0..1)
    """
    # Make sure input requires grad
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    # Create LayerGradCam using the wrapped model and the chosen layer from raw_model
    lgc = LayerGradCam(wrapped_model, target_layer)
    # Forward once to let wrapper pick the detection
    with torch.no_grad():
        _ = wrapped_model(input_tensor)
        score = safe_float(wrapped_model.last_score)
    if score < 0.1:
        return None

    # Compute Grad-CAM: returns attribution on target layer activation shape [B, C, Hf, Wf]
    # LayerGradCam.attribute expects target (int) or None if wrapped_model.forward returns scalar
    cam = lgc.attribute(input_tensor, target=None)  # shape: [1, C, Hf, Wf] or [1, Hf, Wf] depending
    # Convert to numpy heatmap by aggregating channels
    cam_np = cam.detach().cpu().numpy()[0]
    if cam_np.ndim == 3:
        heat = np.sum(np.abs(cam_np), axis=0)
    else:
        heat = np.abs(cam_np)
    # normalize
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / (np.percentile(heat, 99) + 1e-9)
    # Upsample to model input size
    heat_up = cv2.resize(heat, (model_size, model_size), interpolation=cv2.INTER_CUBIC)
    return heat_up

# ----------------- Fallback to IG if no conv found -----------------
def ig_fallback(wrapped_model, input_tensor, baseline=None, device='cpu', nt_samples=5, internal_batch_size=1):
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    ig = IntegratedGradients(wrapped_model)
    nt = NoiseTunnel(ig)
    if baseline is None:
        # blurred baseline
        img_np = input_tensor.detach().cpu().numpy()[0].transpose(1,2,0)
        img_blur = cv2.GaussianBlur((img_np*255).astype(np.uint8), (51,51), 0).astype(np.float32)/255.0
        baseline = torch.from_numpy(img_blur.transpose(2,0,1)).unsqueeze(0).float().to(device)
    attr = nt.attribute(input_tensor, baselines=baseline, nt_type='smoothgrad_sq', nt_samples=max(3, nt_samples), stdevs=0.1, internal_batch_size=internal_batch_size)
    attr_np = attr.detach().cpu().numpy()[0]
    heat = np.sum(np.abs(attr_np), axis=0)
    heat[:6,:6] = 0
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / (np.percentile(heat, 99) + 1e-9)
    heat_up = cv2.resize(heat, (baseline.shape[-1] if False else 640, baseline.shape[-1] if False else 640), interpolation=cv2.INTER_CUBIC)
    return heat_up

# ----------------- Model load (robust) -----------------
def load_model_safe(weights_path, device):
    model = None
    # Try ultralytics loader
    try:
        from ultralytics import YOLO
        print("Loading model via ultralytics.YOLO(...)")
        model = YOLO(weights_path).model
        model = model.float().to(device).eval()
        return model
    except Exception as e:
        print("ultralytics loader failed:", e)
    # Fallback to torch.load with allowlist if available
    try:
        ckpt = None
        if hasattr(torch.serialization, "add_safe_globals"):
            try:
                from ultralytics.nn.tasks import DetectionModel
                with torch.serialization.add_safe_globals([DetectionModel]):
                    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            except Exception:
                with torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"]):
                    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        else:
            ckpt = torch.load(weights_path, map_location=device)
        model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model = model.float().to(device).eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-size', type=int, default=640)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = load_model_safe(args.weights, device)
    wrapped = DEYOWrapper(model)

    # find target conv layer
    conv_info = find_last_conv(model)
    if conv_info is None:
        print("Warning: no Conv2d layer found in model. Will fallback to IG.")
        target_layer = None
    else:
        print(f"Using target conv layer: {conv_info[0]}")
        target_layer = conv_info[1]

    # prepare folders
    if not os.path.exists(args.source):
        print("Input folder not found.")
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.webp']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(args.source, e)))
    print(f"Found {len(files)} images.")

    for img_path in tqdm(sorted(files)):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.out, f"gradcam_{fname}")
        img_t, img0 = preprocess_image(img_path, img_size=args.model_size, device=device)
        if img_t is None:
            continue
        img_t.requires_grad_(True)

        # quick score check
        with torch.no_grad():
            _ = wrapped(img_t)
            score = safe_float(wrapped.last_score)
            cls_id = safe_int(wrapped.last_class_id)
        if score < 0.2:
            gray_ver = (img0 * 0.3).astype(np.uint8)
            cv2.putText(gray_ver, "NO DETECTION", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imwrite(out_path, gray_ver)
            continue

        heatmap = None
        if target_layer is not None:
            try:
                heatmap = gradcam_for_image(wrapped, model, img_t, target_layer, device, model_size=args.model_size)
            except Exception as e:
                print("Grad-CAM failed, fallback IG:", e)
                heatmap = None

        if heatmap is None:
            try:
                heatmap = ig_fallback(wrapped, img_t, device=device, nt_samples=args.samples, internal_batch_size=args.batch_size)
            except Exception as e:
                print("IG fallback failed:", e)
                heatmap = None

        if heatmap is not None:
            try:
                out_img = create_thermal_overlay(img0, heatmap)
                cv2.putText(out_img, f"CONF: {score:.2f} | Class: {cls_id}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
                cv2.imwrite(out_path, out_img)
            except Exception as e:
                print("Visualization failed:", e)
                gray = (img0 * 0.3).astype(np.uint8)
                cv2.putText(gray, "VIS FAIL", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imwrite(out_path, gray)
        else:
            gray = (img0 * 0.3).astype(np.uint8)
            cv2.putText(gray, "NO HEATMAP", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imwrite(out_path, gray)

    print("Done. Results:", args.out)

if __name__ == "__main__":
    main()
