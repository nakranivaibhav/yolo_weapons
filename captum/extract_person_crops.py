import sys
from pathlib import Path
import cv2
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEYO_ROOT = PROJECT_ROOT / "DEYO"
sys.path.insert(0, str(DEYO_ROOT))

from ultralytics import RTDETR

def main():
    parser = argparse.ArgumentParser(description='Extract exactly 500 person crops from entire video')
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "captum" / "crops"), help="Output folder for crops")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"))
    parser.add_argument("--conf", type=float, default=0.3, help="Person detection confidence")
    parser.add_argument("--expand", type=float, default=0.15, help="ROI expansion ratio")
    parser.add_argument("--target_crops", type=int, default=500, help="Target number of crops")
    parser.add_argument("--skip", type=int, default=5, help="Initial frame skip (auto-adjusts)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Target: {args.target_crops} crops")
    print(f"📦 Loading DEYO model: {args.model}")
    model = RTDETR(args.model)
    print("✅ Model loaded\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")
    print(f"⏱️  Duration: {total_frames/fps:.1f} seconds\n")

    # ======================================================
    # PASS 1: Count total detections across entire video
    # ======================================================
    print("🔍 Pass 1: Scanning video to count detections...")
    
    all_detections = []  # Store (frame_idx, box_data) for all detections
    frame_idx = 0
    skip = args.skip
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        
        results = model(frame, conf=args.conf, verbose=False, classes=[0])[0]
        
        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Store detection info
            all_detections.append({
                'frame_idx': frame_idx,
                'box_idx': i,
                'conf': conf,
                'bbox': (x1, y1, x2, y2)
            })
        
        if frame_idx % 100 == 0:
            print(f"  Scanned frame {frame_idx}/{total_frames} ({len(all_detections)} detections so far)", end='\r')
        
        frame_idx += 1
    
    print(f"\n✅ Found {len(all_detections)} total detections across {frame_idx} frames\n")
    
    if len(all_detections) == 0:
        print("❌ No persons detected in video!")
        cap.release()
        return
    
    # ======================================================
    # PASS 2: Smart sampling to get exactly target_crops
    # ======================================================
    
    target_crops = args.target_crops
    
    if len(all_detections) <= target_crops:
        # Use all detections
        print(f"ℹ️  Total detections ({len(all_detections)}) <= target ({target_crops})")
        print(f"   Will extract all {len(all_detections)} crops\n")
        selected_detections = all_detections
    else:
        # Sample evenly across video
        print(f"📊 Sampling {target_crops} crops from {len(all_detections)} detections...")
        
        # Sort by frame index for temporal distribution
        all_detections.sort(key=lambda x: x['frame_idx'])
        
        # Calculate sampling interval
        interval = len(all_detections) / target_crops
        
        # Sample evenly
        selected_detections = []
        for i in range(target_crops):
            idx = int(i * interval)
            selected_detections.append(all_detections[idx])
        
        print(f"✅ Selected {len(selected_detections)} crops evenly distributed\n")
    
    # ======================================================
    # PASS 3: Extract the selected crops
    # ======================================================
    
    print(f"✂️  Extracting crops...")
    
    # Group by frame for efficient extraction
    from collections import defaultdict
    detections_by_frame = defaultdict(list)
    for det in selected_detections:
        detections_by_frame[det['frame_idx']].append(det)
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    crop_count = 0
    current_frame_idx = 0
    frames_to_extract = set(detections_by_frame.keys())
    
    while frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame_idx in frames_to_extract:
            # Extract crops from this frame
            for det in detections_by_frame[current_frame_idx]:
                x1, y1, x2, y2 = det['bbox']
                
                # Expand ROI
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
                
                # Save with unique name
                crop_path = out_dir / f"frame{det['frame_idx']:05d}_person{det['box_idx']:02d}_conf{det['conf']:.2f}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crop_count += 1
            
            frames_to_extract.remove(current_frame_idx)
            
            if crop_count % 50 == 0:
                print(f"  Extracted {crop_count}/{len(selected_detections)} crops", end='\r')
        
        current_frame_idx += 1
    
    cap.release()
    
    print(f"\n\n✅ Done! Saved {crop_count} crops to {out_dir}")
    print(f"📂 Output: {out_dir.absolute()}")
    
    # Print distribution info
    if crop_count > 0:
        frame_indices = [det['frame_idx'] for det in selected_detections[:crop_count]]
        print(f"\n📊 Distribution:")
        print(f"   First crop: frame {min(frame_indices)}")
        print(f"   Last crop:  frame {max(frame_indices)}")
        print(f"   Span:       {max(frame_indices) - min(frame_indices)} frames")

if __name__ == "__main__":
    main()