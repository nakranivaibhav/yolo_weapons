import torch
import cv2
import numpy as np
import argparse
from ultralytics.data.augment import LetterBox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt file')
    parser.add_argument('--source', type=str, required=True, help='Path to video file')
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model: {args.weights}...")
    ckpt = torch.load(args.weights, weights_only=False)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model.float().eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 2. Print Model Classes (The Answer Key)
    if hasattr(model, 'names'):
        print("\n📋 MODEL CLASS MAPPING:")
        print(model.names)
    else:
        print("\n⚠️ Model has no '.names' attribute.")

    # 3. Load First Frame
    cap = cv2.VideoCapture(args.source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Failed to read video.")
        return

    # 4. Preprocess (LetterBox)
    lb = LetterBox(640, auto=False, stride=32)
    img = lb(image=frame)
    img = img.transpose((2, 0, 1))[::-1] 
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to(device)
    img /= 255.0
    img = img.unsqueeze(0)

    # 5. Run Inference
    preds = model(img)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    
    # Extract logits: [Batch, Queries, 4+Classes]
    # We skip the first 4 columns (bbox coords)
    logits = preds[0, :, 4:]
    probs = logits.softmax(dim=-1)

    # 6. Find the ACTUAL detections
    # Get the max score for EACH query (box)
    scores, class_ids = probs.max(dim=-1)
    
    # Filter for anything confident
    mask = scores > 0.2
    confident_scores = scores[mask]
    confident_classes = class_ids[mask]

    print("\n🔎 WHAT THE MODEL SEES IN FRAME 0:")
    if len(confident_scores) == 0:
        print("❌ Nothing! The model detects NOTHING with >20% confidence.")
        print("   -> Try a different video or check if your weights are loaded correctly.")
        print(f"   -> Max score found in entire image: {scores.max().item():.4f}")
    else:
        for i in range(len(confident_scores)):
            cls_id = confident_classes[i].item()
            score = confident_scores[i].item()
            name = model.names[cls_id] if hasattr(model, 'names') else "???"
            print(f"   ✅ Detected: '{name}' (ID: {cls_id}) | Confidence: {score:.4f}")

    print("-" * 40)

if __name__ == "__main__":
    main()