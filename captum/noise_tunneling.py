import torch
import cv2
import numpy as np
import argparse
import sys
import time
from captum.attr import IntegratedGradients, NoiseTunnel
from ultralytics.data.augment import LetterBox

# --- 1. The Wrapper ---
class DEYOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        preds = self.model(input_tensor)
        
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        # Extract logits. 
        # For single class model, shape is [Batch, Queries, 5] (4 box + 1 score)
        if isinstance(preds, dict) and 'pred_logits' in preds:
             logits = preds['pred_logits'][0]
        else:
            logits = preds[0, :, 4:] 
        
        probs = logits.sigmoid() # Use Sigmoid for single-class often, or Softmax if 2 classes (bg/fg)
        
        # Target Class 0 (The ONLY class)
        target_class_idx = 0 
        best_query_idx = probs[:, target_class_idx].argmax()
        
        # DEBUG: Store score to check later
        self.last_score = probs[best_query_idx, target_class_idx]

        # Safety: If confidence is low, return zero gradient (disconnected graph)
        if self.last_score < 0.2:
            return (input_tensor * 0).sum().view(1)

        return logits[best_query_idx, target_class_idx].unsqueeze(0)

# --- 2. Robust Normalization ---
def robust_normalize_heatmap(attr_np):
    if np.max(attr_np) == 0:
        return np.zeros_like(attr_np, dtype=np.uint8)
    # Clip outliers (top 1%) to make the main heatmap brighter
    v_max = np.percentile(attr_np, 99)
    attr_np = np.clip(attr_np, 0, v_max)
    norm = attr_np / (v_max + 1e-8)
    return np.uint8(255 * norm)

def preprocess_frame(img0, img_size=640, device='cpu'):
    # Use LetterBox for correct aspect ratio
    lb = LetterBox(img_size, auto=False, stride=32)
    img = lb(image=img0)
    
    img_t = img.transpose((2, 0, 1))[::-1] 
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).float()
    img_t /= 255.0
    if len(img_t.shape) == 3:
        img_t = img_t[None]
    return img_t.to(device), img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--out', type=str, default='gun_heatmap.mp4')
    parser.add_argument('--limit', type=int, default=100, help="Frames to process AFTER finding gun")
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on: {device}")

    print(f"Loading model: {args.weights}...")
    ckpt = torch.load(args.weights, weights_only=False)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model.float().to(device).eval()

    wrapped_model = DEYOWrapper(model)
    ig = IntegratedGradients(wrapped_model)
    nt = NoiseTunnel(ig)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None # Init later

    frame_count = 0
    processed_count = 0
    found_gun = False
    
    print(f"Scanning video for Gun (Conf > 0.2)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Stop if we hit our processing limit
        if processed_count >= args.limit:
            print("\nLimit reached.")
            break

        # Preprocess
        input_tensor, resized_img = preprocess_frame(frame, device=device)
        input_tensor.requires_grad = True

        # --- AUTO-SEEK LOGIC ---
        # Run a quick check (without gradients) to see if a gun is here
        with torch.no_grad():
            _ = wrapped_model(input_tensor)
            score = wrapped_model.last_score.item()
        
        if not found_gun:
            if score < 0.2:
                print(f"Skipping Frame {frame_count} (Conf: {score:.2f})...", end='\r')
                frame_count += 1
                continue
            else:
                print(f"\n✅ FOUND GUN at Frame {frame_count}! (Conf: {score:.2f}) Starting analysis...")
                found_gun = True
                # Init Video Writer now that we have data
                h, w = resized_img.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # --- CAPTUM ANALYSIS ---
        try:
            attributions = nt.attribute(
                input_tensor,
                nt_type='smoothgrad_sq',
                nt_samples=args.samples,
                stdevs=0.15, 
                internal_batch_size=args.samples
            )

            attr_np = attributions.detach().cpu().numpy()[0]
            attr_np = np.sum(np.abs(attr_np), axis=0)
            
            heatmap_uint8 = robust_normalize_heatmap(attr_np)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
            
            # Overlay
            overlay = cv2.addWeighted(resized_img, 0.6, heatmap_color, 0.4, 0)
            
            # Add Text
            cv2.putText(overlay, f"Gun Conf: {score:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(overlay)
            
            processed_count += 1
            frame_count += 1
            print(f"Generated Heatmap {processed_count}/{args.limit} | Frame: {frame_count} | Conf: {score:.2f}", end='\r')

        except Exception as e:
            print(f"\nError: {e}")
            break

    cap.release()
    if writer: writer.release()
    print(f"\nDone! Output saved to {args.out}")

if __name__ == "__main__":
    main()