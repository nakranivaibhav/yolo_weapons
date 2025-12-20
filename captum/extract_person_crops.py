import sys
from pathlib import Path
import cv2
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEYO_ROOT = PROJECT_ROOT / "DEYO"
sys.path.insert(0, str(DEYO_ROOT))

from ultralytics import RTDETR

def main():
    parser = argparse.ArgumentParser(description='Extract person crops from video')
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "captum" / "crops"), help="Output folder for crops")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"))
    parser.add_argument("--conf", type=float, default=0.3, help="Person detection confidence")
    parser.add_argument("--expand", type=float, default=0.15, help="ROI expansion ratio")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--skip", type=int, default=5, help="Process every N-th frame")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DEYO model: {args.model}")
    model = RTDETR(args.model)
    print("Model loaded\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {total_frames} frames")
    print(f"Processing every {args.skip} frames, max {args.max_frames} frames\n")

    crop_count = 0
    frame_idx = 0
    processed = 0

    while processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        results = model(frame, conf=args.conf, verbose=False, classes=[0])[0]

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            w = x2 - x1
            h = y2 - y1
            expand_w = int(w * args.expand)
            expand_h = int(h * args.expand)
            
            x1_exp = max(0, x1 - expand_w)
            y1_exp = max(0, y1 - expand_h)
            x2_exp = min(width, x2 + expand_w)
            y2_exp = min(height, y2 + expand_h)
            
            crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            
            if crop.size == 0:
                continue
            
            crop_path = out_dir / f"frame{frame_idx:05d}_person{i:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_count += 1

        if processed % 10 == 0:
            print(f"Frame {frame_idx}: {len(results.boxes)} persons")

        frame_idx += 1
        processed += 1

    cap.release()
    
    print(f"\nDone! Saved {crop_count} crops to {out_dir}")

if __name__ == "__main__":
    main()
