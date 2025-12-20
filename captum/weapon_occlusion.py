import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import argparse
import glob
from tqdm import tqdm
from captum.attr import Occlusion

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model, target_pred_idx=None, target_cls=None):
        super().__init__()
        self.model = model
        self.target_pred_idx = target_pred_idx
        self.target_cls = target_cls

    def forward(self, x):
        batch_size = x.shape[0]
        
        with torch.enable_grad():
            output = self.model(x)
        
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        if output.dim() == 2:
            output = output.unsqueeze(0)
        
        results = []
        for i in range(batch_size):
            preds = output[i]
            
            if preds.shape[0] < preds.shape[1]:
                pass
            else:
                preds = preds.T
            
            class_scores = preds[4:, :]
            
            if self.target_pred_idx is not None and self.target_cls is not None:
                score = class_scores[self.target_cls, self.target_pred_idx]
            else:
                conf, _ = class_scores.max(dim=0)
                best_idx = conf.argmax()
                score = conf[best_idx]
            
            results.append(score)
        
        return torch.stack(results).unsqueeze(1)

def get_detection(yolo, img, conf_thresh, size):
    results = yolo(img, imgsz=size, conf=conf_thresh, verbose=False)[0]
    
    if len(results.boxes) == 0:
        return None
    
    boxes = results.boxes
    best_idx = boxes.conf.argmax()
    
    box = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx])
    cls = int(boxes.cls[best_idx])
    
    return {
        'box': box,
        'conf': conf,
        'cls': cls,
        'label': results.names[cls]
    }

def find_prediction_index(model, tensor, detection, device):
    with torch.no_grad():
        output = model(tensor)
        
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        preds = output[0]
        if preds.shape[0] > preds.shape[1]:
            preds = preds.T
        
        boxes = preds[:4, :].T
        class_scores = preds[4:, :]
        
        target_cls = detection['cls']
        cls_scores = class_scores[target_cls, :]
        
        best_idx = cls_scores.argmax().item()
        best_score = float(cls_scores[best_idx])
        
        return best_idx, best_score

def preprocess(img, size=640, device='cpu'):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return tensor.to(device), (dh, dw, nh, nw, h, w)

def overlay_heatmap(img, heatmap, pad_info, size=640):
    dh, dw, nh, nw, orig_h, orig_w = pad_info
    
    heatmap_up = cv2.resize(heatmap, (size, size))
    heatmap_crop = heatmap_up[dh:dh+nh, dw:dw+nw]
    
    if heatmap_crop.size == 0:
        heatmap_crop = heatmap_up
    
    heatmap_resized = cv2.resize(heatmap_crop, (orig_w, orig_h))
    heatmap_blur = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    
    if heatmap_blur.max() > 0:
        heatmap_norm = heatmap_blur / (np.percentile(heatmap_blur, 99) + 1e-8)
        heatmap_norm = np.clip(heatmap_norm, 0, 1)
    else:
        heatmap_norm = np.zeros_like(heatmap_blur)
    
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img, 0.65, heatmap_color, 0.35, 0)
    return overlay

def create_side_by_side(img, overlay, label, conf):
    h, w = img.shape[:2]
    
    target_h = 400
    scale = target_h / h
    new_w = int(w * scale)
    new_h = target_h
    
    img_resized = cv2.resize(img, (new_w, new_h))
    overlay_resized = cv2.resize(overlay, (new_w, new_h))
    
    gap = 10
    canvas = np.full((new_h + 50, new_w * 2 + gap, 3), 40, dtype=np.uint8)
    
    canvas[40:40+new_h, 0:new_w] = img_resized
    canvas[40:40+new_h, new_w+gap:new_w*2+gap] = overlay_resized
    
    cv2.putText(canvas, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Occlusion - {label}: {conf:.2f}", (new_w + gap + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description='Occlusion attribution for YOLO weapon detection')
    parser.add_argument("--crops", type=str, default=str(PROJECT_ROOT / "captum" / "crops"))
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "captum" / "occlusion_output"))
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "weapon_detection_yolo11m_640" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--window", type=int, default=32, help="Occlusion window size")
    parser.add_argument("--stride", type=int, default=16, help="Stride between windows")
    parser.add_argument("--batch", type=int, default=32, help="Perturbations per batch")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {args.model}")
    from ultralytics import YOLO
    yolo = YOLO(args.model)
    model = yolo.model.float().to(device)
    model.eval()
    
    print(f"Model classes: {yolo.names}")

    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(str(Path(args.crops) / ext)))
    images = sorted(images)
    print(f"Found {len(images)} images")
    print(f"Window: {args.window}x{args.window}, Stride: {args.stride}\n")

    detection_count = 0
    
    for img_path in tqdm(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        fname = Path(img_path).stem
        
        detection = get_detection(yolo, img, args.conf, args.size)
        
        if detection is None:
            out_path = out_dir / f"{fname}_no_detection.jpg"
            gray = (img * 0.5).astype(np.uint8)
            side_by_side = create_side_by_side(img, gray, "NO WEAPON", 0.0)
            cv2.imwrite(str(out_path), side_by_side)
            continue

        tensor, pad_info = preprocess(img, args.size, device)
        
        pred_idx, raw_score = find_prediction_index(model, tensor, detection, device)
        
        wrapper = YOLOWrapper(model, target_pred_idx=pred_idx, target_cls=detection['cls'])
        occlusion = Occlusion(wrapper)

        try:
            attr = occlusion.attribute(
                tensor,
                sliding_window_shapes=(3, args.window, args.window),
                strides=(3, args.stride, args.stride),
                baselines=0,
                perturbations_per_eval=args.batch,
                show_progress=False
            )
            
            heatmap = attr.detach().cpu().numpy()[0]
            heatmap = np.abs(heatmap).mean(axis=0)
            
            overlay = overlay_heatmap(img, heatmap, pad_info, args.size)
            
            side_by_side = create_side_by_side(img, overlay, detection['label'], detection['conf'])
            
            out_path = out_dir / f"{fname}_{detection['label']}.jpg"
            cv2.imwrite(str(out_path), side_by_side)
            detection_count += 1
            
        except Exception as e:
            print(f"Occlusion failed for {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! {detection_count} weapons detected")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()
