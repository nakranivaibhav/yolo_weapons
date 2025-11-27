import sys
from pathlib import Path
import cv2
import numpy as np
import time
import subprocess
import json
import base64

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEYO_ROOT = PROJECT_ROOT / "DEYO"

import argparse
parser = argparse.ArgumentParser(description='Simple DEYO Person + ONNX Weapon Detection')
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--deyo_model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"))
parser.add_argument("--weapon_model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "weapon_detection_yolo11m_640" / "weights" / "best.pt"))
parser.add_argument("--person_conf", type=float, default=0.3)
parser.add_argument("--weapon_conf", type=float, default=0.25)
parser.add_argument("--iou", type=float, default=0.45)
parser.add_argument("--roi_expand", type=float, default=0.15)
parser.add_argument("--downscale", type=float, default=0.5)
parser.add_argument("--max_frames", type=int, default=60)
parser.add_argument("--track", action="store_true", help="Enable ByteTrack for temporal filtering")
parser.add_argument("--track_persist", type=int, default=30, help="Frames to persist track after disappearance")
parser.add_argument("--min_hits", type=int, default=3, help="Minimum hits before track is confirmed")
parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "inference_output" / "person_weapon_simple.mp4"))
args = parser.parse_args()

print(f"\n{'='*80}")
print("SIMPLE PERSON + WEAPON DETECTION (Sequential)")
print(f"{'='*80}\n")

print("[1/2] Loading DEYO person model...")
sys.path.insert(0, str(DEYO_ROOT))
from ultralytics import RTDETR
person_model = RTDETR(args.deyo_model)
print("      ✓ DEYO loaded")

print("[2/2] Loading YOLO weapon model (subprocess with GPU)...")
weapon_subprocess = subprocess.Popen(
    [sys.executable, str(PROJECT_ROOT / "inference" / "weapon_detector_subprocess.py"), 
     args.weapon_model, str(args.weapon_conf)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

while True:
    line = weapon_subprocess.stderr.readline()
    if "WEAPON_MODEL_READY" in line:
        break
    if not line:
        raise RuntimeError("Weapon model subprocess failed to start")

WEAPON_LABELS = {0: 'knife', 1: 'gun', 2: 'rifle', 3: 'baseball_bat'}
print(f"      ✓ YOLO weapon model loaded on GPU (subprocess)\n")

if args.track:
    import logging
    logging.getLogger('boxmot').setLevel(logging.WARNING)
    from boxmot import ByteTrack
    
    tracker = ByteTrack(
        track_thresh=args.weapon_conf,
        track_buffer=args.track_persist,
        match_thresh=0.8,
        frame_rate=30,
        min_hits=args.min_hits
    )
    print(f"[INFO] ByteTrack enabled (min_hits={args.min_hits}, persist={args.track_persist} frames)\n")
else:
    tracker = None

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width_person = int(width_orig * args.downscale)
height_person = int(height_orig * args.downscale)
scale_x = width_orig / width_person
scale_y = height_orig / height_person

print(f"Video: {width_orig}x{height_orig} @ {fps} FPS")
print(f"Person detection: {width_person}x{height_person}")
print(f"Weapon detection: Full-res crops\n")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out_video = cv2.VideoWriter(args.out, fourcc, fps, (width_person, height_person))

stats = {
    'person_count': [],
    'weapon_count': [],
    'person_times': [],
    'weapon_times': []
}

print(f"Processing {args.max_frames} frames...\n")

for frame_idx in range(args.max_frames):
    ret, frame_orig = cap.read()
    if not ret:
        break
    
    frame_person = cv2.resize(frame_orig, (width_person, height_person))
    
    t_start = time.perf_counter()
    person_results = person_model(frame_person, conf=args.person_conf, verbose=False, classes=[0])[0]
    t_person = (time.perf_counter() - t_start) * 1000
    
    person_boxes = []
    for box in person_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        person_boxes.append([float(x1), float(y1), float(x2), float(y2), conf])
    
    weapon_detections = []
    t_weapon = 0
    
    if len(person_boxes) > 0:
        t_start = time.perf_counter()
        
        for person_box in person_boxes:
            x1_p, y1_p, x2_p, y2_p, conf_p = person_box
            
            x1_full = int(x1_p * scale_x)
            y1_full = int(y1_p * scale_y)
            x2_full = int(x2_p * scale_x)
            y2_full = int(y2_p * scale_y)
            
            w = x2_full - x1_full
            h = y2_full - y1_full
            expand_w = int(w * args.roi_expand)
            expand_h = int(h * args.roi_expand)
            
            x1_exp = max(0, x1_full - expand_w)
            y1_exp = max(0, y1_full - expand_h)
            x2_exp = min(width_orig, x2_full + expand_w)
            y2_exp = min(height_orig, y2_full + expand_h)
            
            person_crop = frame_orig[y1_exp:y2_exp, x1_exp:x2_exp]
            
            if person_crop.size == 0:
                continue
            
            crop_h, crop_w = person_crop.shape[:2]
            max_dim = max(crop_h, crop_w)
            imgsz = 640
            
            crop_bytes = person_crop.tobytes()
            crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
            
            request = json.dumps({
                'crop': crop_b64,
                'shape': person_crop.shape,
                'imgsz': imgsz
            })
            
            weapon_subprocess.stdin.write(request + '\n')
            weapon_subprocess.stdin.flush()
            
            response = weapon_subprocess.stdout.readline()
            detections = json.loads(response.strip())
            
            for det in detections:
                x1_w, y1_w, x2_w, y2_w, conf_w, cls_w = det
                
                x1_global = int(x1_w + x1_exp)
                y1_global = int(y1_w + y1_exp)
                x2_global = int(x2_w + x1_exp)
                y2_global = int(y2_w + y1_exp)
                
                x1_scaled = x1_global / scale_x
                y1_scaled = y1_global / scale_y
                x2_scaled = x2_global / scale_x
                y2_scaled = y2_global / scale_y
                
                weapon_detections.append({
                    'weapon': [x1_scaled, y1_scaled, x2_scaled, y2_scaled, conf_w, cls_w],
                    'person': [x1_p, y1_p, x2_p, y2_p, conf_p]
                })
        
        t_weapon = (time.perf_counter() - t_start) * 1000
    
    if args.track and tracker is not None and len(weapon_detections) > 0:
        dets_np = np.array([[d['weapon'][0] * scale_x, d['weapon'][1] * scale_y, 
                             d['weapon'][2] * scale_x, d['weapon'][3] * scale_y, 
                             d['weapon'][4], d['weapon'][5]] for d in weapon_detections])
        
        tracks = tracker.update(dets_np, frame_orig)
        
        tracked_detections = []
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                tracked_detections.append({
                    'weapon': [x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y, conf, cls],
                    'track_id': int(track_id)
                })
        weapon_detections = tracked_detections
    elif args.track and tracker is not None:
        tracker.update(np.empty((0, 6)), frame_orig)
    
    stats['person_count'].append(len(person_boxes))
    stats['weapon_count'].append(len(weapon_detections))
    stats['person_times'].append(t_person)
    stats['weapon_times'].append(t_weapon)
    
    annotated = frame_person.copy()
    
    for det in weapon_detections:
        x1_w, y1_w, x2_w, y2_w, conf_w, cls_w = det['weapon']
        weapon_name = WEAPON_LABELS[int(cls_w)]
        color = (0, 0, 255) if weapon_name in ['gun', 'rifle'] else (0, 255, 255)
        
        cv2.rectangle(annotated, (int(x1_w), int(y1_w)), (int(x2_w), int(y2_w)), color, 3)
        
        if args.track and 'track_id' in det:
            label = f"ID:{det['track_id']} {weapon_name} {conf_w:.2f}"
        else:
            label = f"{weapon_name} {conf_w:.2f}"
        
        cv2.putText(annotated, label, (int(x1_w), int(y1_w)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    out_video.write(annotated)
    
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}: {len(person_boxes)} persons, {len(weapon_detections)} weapons | "
              f"Person: {t_person:.1f}ms, Weapon: {t_weapon:.1f}ms")

cap.release()
out_video.release()

weapon_subprocess.stdin.close()
weapon_subprocess.wait()

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"Frames processed: {len(stats['person_count'])}")
print(f"Total persons detected: {sum(stats['person_count'])}")
print(f"Total weapons detected: {sum(stats['weapon_count'])}")
print(f"Avg person detection time: {np.mean(stats['person_times']):.1f}ms")
if any(t > 0 for t in stats['weapon_times']):
    print(f"Avg weapon detection time: {np.mean([t for t in stats['weapon_times'] if t > 0]):.1f}ms")
print(f"\n📹 Output: {args.out}\n")

