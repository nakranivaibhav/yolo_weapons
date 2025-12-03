import torch
import cv2
import numpy as np
import argparse
import sys
from captum.attr import LayerGradCam
from ultralytics.data.augment import LetterBox

# --- 1. The Wrapper (Same as before) ---
class DEYOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        # Run model
        preds = self.model(input_tensor)
        
        # 1. Handle List/Tuple Output
        # RT-DETR usually returns a tuple where the first element is the prediction tensor
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        # 2. Extract Logits from Concatenated Tensor
        # The tensor shape is usually [Batch, 300, 4 + NumClasses]
        # First 4 columns = Bounding Box coordinates (x,y,w,h)
        # Remaining columns = Class Scores (Logits)
        
        # Check if it's a dictionary (just in case)
        if isinstance(preds, dict) and 'pred_logits' in preds:
             logits = preds['pred_logits'][0]
        else:
            # Assume concatenated tensor [Batch, Queries, 4+Classes]
            # We slice from index 4 to the end to get the scores
            logits = preds[0, :, 4:] 
        
        probs = logits.softmax(dim=-1)
        
        # Target Class: 0 (Gun) - Change this if your gun class ID is different!
        target_class_idx = 0
        
        # Find the single box with the highest score for 'Gun'
        best_query_idx = probs[:, target_class_idx].argmax()
        
        # Return that SINGLE score for Captum to trace
        # We need to return it as a 1D tensor with 1 element
        return logits[best_query_idx, target_class_idx].unsqueeze(0)

# --- 2. Frame Preprocessing ---
def preprocess_frame(img0, img_size=640):
    """
    Prepares a single video frame for the model.
    """
    # Resize with LetterBox (Padding) to keep aspect ratio
    img = LetterBox(img_size, auto=False, stride=32)(image=img0)
    
    # Convert to Tensor
    img_t = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).float()
    img_t /= 255.0  
    if len(img_t.shape) == 3:
        img_t = img_t[None]  
        
    return img_t, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--source', type=str, required=True, help='Path to input video.mp4')
    parser.add_argument('--out', type=str, default='gradcam_output.mp4', help='Output video name')
    parser.add_argument('--limit', type=int, default=300, help='Limit processing to N frames (default 300)')
    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading model: {args.weights}...")
    #ckpt = torch.load(args.weights)
    # explicitly allow loading the full model structure
    ckpt = torch.load(args.weights, weights_only=False)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model.float()
    model.eval()

    # --- Select Layer ---
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    print(f"Targeting Layer: {target_layer}")

    # --- Init GradCAM ---
    wrapped_model = DEYOWrapper(model)
    lgc = LayerGradCam(wrapped_model, target_layer)

    # --- Setup Video ---
    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height} @ {fps}FPS ({total_frames} frames)")

    # Output is resized to 640x640 (Model Size) to avoid alignment bugs
    out_size = (640, 640)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= args.limit:
            break

        # 1. Preprocess
        input_tensor, resized_img = preprocess_frame(frame)
        input_tensor.requires_grad = True

        try:
            # 2. Compute Attribution
            attributions = lgc.attribute(input_tensor)
            
            # 3. Process Heatmap (No Matplotlib!)
            # Upsample to 640x640
            attr_up = LayerGradCam.interpolate(attributions, (640, 640))
            attr_np = attr_up.detach().cpu().numpy()[0, 0] # [640, 640]

            # Normalize to 0-255
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
            attr_uint8 = np.uint8(255 * attr_np)

            # Apply Colormap (Jet = Blue to Red)
            heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)

            # 4. Overlay
            # Blend: 60% Original + 40% Heatmap
            overlay = cv2.addWeighted(resized_img, 0.6, heatmap, 0.4, 0)

            # Write frame
            writer.write(overlay)
            
            print(f"Processed frame {frame_count+1}/{args.limit}", end='\r')
            frame_count += 1
            
        except Exception as e:
            print(f"\nError on frame {frame_count}: {e}")
            break

    cap.release()
    writer.release()
    print(f"\nDone! Saved to {args.out}")

if __name__ == "__main__":
    main()