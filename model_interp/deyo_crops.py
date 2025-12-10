"""
Extract and save person bounding box crops from DEYO model detections.
Processes an MP4 video and saves cropped images to disk.
"""
import sys
from pathlib import Path
import cv2
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEYO_ROOT = PROJECT_ROOT / "DEYO"

parser = argparse.ArgumentParser(description='Extract DEYO person bounding box crops from video')
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--deyo_model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"),
                    help="Path to DEYO model weights")
parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "inference_output" / "deyo_crops"),
                    help="Directory to save cropped images")
parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for person detection")
parser.add_argument("--downscale", type=float, default=1.0, 
                    help="Downscale factor for inference (1.0 = full resolution)")
parser.add_argument("--roi_expand", type=float, default=0.0,
                    help="Expand bounding box by this fraction (0.0 = no expansion)")
parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (None = all)")
parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame")
parser.add_argument("--save_format", type=str, default="jpg", choices=["jpg", "png"],
                    help="Image format for saved crops")
args = parser.parse_args()

print(f"\n{'='*80}")
print("DEYO PERSON CROP EXTRACTOR")
print(f"{'='*80}\n")

# Load DEYO model
print("[1/2] Loading DEYO person model...")
sys.path.insert(0, str(DEYO_ROOT))
from ultralytics import RTDETR
deyo_model = RTDETR(args.deyo_model)
print("      ✓ DEYO loaded")

# Open video
print("[2/2] Opening video...")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width_infer = int(width_orig * args.downscale)
height_infer = int(height_orig * args.downscale)
scale_x = width_orig / width_infer
scale_y = height_orig / height_infer

print(f"      ✓ Video opened: {width_orig}x{height_orig} @ {fps} FPS, {total_frames} frames")
print(f"      Inference resolution: {width_infer}x{height_infer}")

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Output directory: {output_dir}\n")

# Process video
frame_idx = 0
crop_count = 0
max_frames = args.max_frames if args.max_frames else total_frames

print(f"Processing frames (skip={args.frame_skip}, max={max_frames})...\n")

while True:
    ret, frame_orig = cap.read()
    if not ret:
        break
    
    if frame_idx >= max_frames:
        break
    
    # Skip frames if requested
    if frame_idx % args.frame_skip != 0:
        frame_idx += 1
        continue
    
    # Resize for inference if downscaling
    if args.downscale != 1.0:
        frame_infer = cv2.resize(frame_orig, (width_infer, height_infer))
    else:
        frame_infer = frame_orig
    
    # Run DEYO person detection (class 0 = person)
    results = deyo_model(frame_infer, conf=args.conf, verbose=False, classes=[0])[0]
    
    # Extract and save crops
    for box_idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        
        # Scale back to original resolution
        x1_full = int(x1 * scale_x)
        y1_full = int(y1 * scale_y)
        x2_full = int(x2 * scale_x)
        y2_full = int(y2 * scale_y)
        
        # Apply ROI expansion if requested
        if args.roi_expand > 0:
            w = x2_full - x1_full
            h = y2_full - y1_full
            expand_w = int(w * args.roi_expand)
            expand_h = int(h * args.roi_expand)
            x1_full = max(0, x1_full - expand_w)
            y1_full = max(0, y1_full - expand_h)
            x2_full = min(width_orig, x2_full + expand_w)
            y2_full = min(height_orig, y2_full + expand_h)
        
        # Extract crop from original resolution frame
        crop = frame_orig[y1_full:y2_full, x1_full:x2_full]
        
        if crop.size == 0:
            continue
        
        # Save crop
        filename = f"frame{frame_idx:06d}_person{box_idx:02d}_conf{conf:.2f}.{args.save_format}"
        crop_path = output_dir / filename
        cv2.imwrite(str(crop_path), crop)
        crop_count += 1
    
    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx}/{max_frames}: {len(results.boxes)} persons detected, {crop_count} total crops saved")
    
    frame_idx += 1

cap.release()

print(f"\n{'='*80}")
print("EXTRACTION COMPLETE")
print(f"{'='*80}")
print(f"Frames processed: {frame_idx}")
print(f"Total crops saved: {crop_count}")
print(f"Output directory: {output_dir}")
print(f"{'='*80}\n")

