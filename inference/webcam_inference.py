import sys
from pathlib import Path
import cv2
import numpy as np
import time
import subprocess
import json
import base64
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEYO_ROOT = PROJECT_ROOT / "DEYO"

import argparse
parser = argparse.ArgumentParser(description='Webcam Person + Weapon Detection (MPS)')
parser.add_argument("--camera", type=int, default=0, help="Camera index")
parser.add_argument("--deyo_model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"))
parser.add_argument("--weapon_model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "25_dec_2025_yolo11m" / "weights" / "best.pt"))
parser.add_argument("--person_conf", type=float, default=0.3)
parser.add_argument("--weapon_conf", type=float, default=0.25)
parser.add_argument("--roi_expand", type=float, default=0.15)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--track", action="store_true", help="Enable ByteTrack for temporal filtering")
parser.add_argument("--track_persist", type=int, default=30)
parser.add_argument("--min_hits", type=int, default=3)
parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
parser.add_argument("--height", type=int, default=720, help="Camera capture height")
parser.add_argument("--save", type=str, default="", help="Save output to video file")
args = parser.parse_args()

print(f"\n{'='*60}")
print("WEBCAM PERSON + WEAPON DETECTION (MPS)")
print(f"{'='*60}\n")

import torch
if torch.backends.mps.is_available():
    device = "mps"
    print(f"[INFO] Using MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print(f"[WARN] MPS not available, using CPU")

print("[1/2] Loading DEYO person model...")
sys.path.insert(0, str(DEYO_ROOT))
from ultralytics import RTDETR
person_model = RTDETR(args.deyo_model)
print("      ✓ DEYO loaded")

print("[2/2] Loading YOLO weapon model (subprocess)...")
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

print("      ✓ YOLO weapon model loaded (subprocess)")

WEAPON_LABELS = {0: 'knife', 1: 'gun', 2: 'rifle', 3: 'baseball_bat'}

tracker = None
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
    print(f"[INFO] ByteTrack enabled\n")

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {args.camera}")

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera: {actual_width}x{actual_height}")
print(f"Press 'q' to quit\n")

out_video = None
if args.save:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    out_video = cv2.VideoWriter(args.save, fourcc, 30, (actual_width, actual_height))
    print(f"[INFO] Saving to {args.save}\n")

fps_history = deque(maxlen=30)
frame_count = 0

def detect_weapons_in_crop(crop):
    crop_bytes = crop.tobytes()
    crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
    request = json.dumps({
        'crop': crop_b64,
        'shape': list(crop.shape),
        'imgsz': args.imgsz
    })
    weapon_subprocess.stdin.write(request + '\n')
    weapon_subprocess.stdin.flush()
    response = weapon_subprocess.stdout.readline()
    while response:
        line = response.strip()
        if line.startswith('[]') or line.startswith('[['):
            break
        response = weapon_subprocess.stdout.readline()
    return json.loads(response.strip()) if response else []

try:
    while True:
        t_frame_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        t_person_start = time.perf_counter()
        person_results = person_model(frame, conf=args.person_conf, verbose=False, 
                                       classes=[0], imgsz=args.imgsz, device=device)[0]
        t_person = (time.perf_counter() - t_person_start) * 1000
        
        person_boxes = []
        for box in person_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
                if 0 <= x1 < x2 <= actual_width and 0 <= y1 < y2 <= actual_height:
                    person_boxes.append([float(x1), float(y1), float(x2), float(y2), conf])
        
        weapon_detections = []
        t_weapon = 0
        
        if len(person_boxes) > 0:
            t_weapon_start = time.perf_counter()
            
            for px1, py1, px2, py2, pconf in person_boxes:
                w = px2 - px1
                h = py2 - py1
                expand_w = int(w * args.roi_expand)
                expand_h = int(h * args.roi_expand)
                
                x1_exp = max(0, int(px1) - expand_w)
                y1_exp = max(0, int(py1) - expand_h)
                x2_exp = min(actual_width, int(px2) + expand_w)
                y2_exp = min(actual_height, int(py2) + expand_h)
                
                person_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                if person_crop.size == 0:
                    continue
                
                detections = detect_weapons_in_crop(person_crop)
                
                for det in detections:
                    wx1, wy1, wx2, wy2, wconf, wcls = det
                    weapon_detections.append({
                        'weapon': [wx1 + x1_exp, wy1 + y1_exp, wx2 + x1_exp, wy2 + y1_exp, wconf, wcls],
                        'person': [px1, py1, px2, py2, pconf]
                    })
            
            t_weapon = (time.perf_counter() - t_weapon_start) * 1000
        
        if tracker is not None and len(weapon_detections) > 0:
            dets_np = np.array([[d['weapon'][0], d['weapon'][1], d['weapon'][2], 
                                 d['weapon'][3], d['weapon'][4], d['weapon'][5]] 
                                for d in weapon_detections])
            tracks = tracker.update(dets_np, frame)
            
            tracked_detections = []
            if len(tracks) > 0:
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf, cls = track[:7]
                    tracked_detections.append({
                        'weapon': [x1, y1, x2, y2, conf, cls],
                        'track_id': int(track_id)
                    })
            weapon_detections = tracked_detections
        elif tracker is not None:
            tracker.update(np.empty((0, 6)), frame)
        
        annotated = frame.copy()
        
        for det in weapon_detections:
            x1_w, y1_w, x2_w, y2_w, conf_w, cls_w = det['weapon']
            weapon_name = WEAPON_LABELS.get(int(cls_w), 'unknown')
            color = (0, 0, 255) if weapon_name in ['gun', 'rifle'] else (0, 255, 255)
            
            cv2.rectangle(annotated, (int(x1_w), int(y1_w)), (int(x2_w), int(y2_w)), color, 3)
            
            if args.track and 'track_id' in det:
                label = f"ID:{det['track_id']} {weapon_name} {conf_w:.2f}"
            else:
                label = f"{weapon_name} {conf_w:.2f}"
            
            cv2.putText(annotated, label, (int(x1_w), int(y1_w)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        t_frame = time.perf_counter() - t_frame_start
        fps = 1.0 / t_frame if t_frame > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        info_text = f"FPS: {avg_fps:.1f} | Persons: {len(person_boxes)} | Weapons: {len(weapon_detections)}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        timing_text = f"Person: {t_person:.0f}ms | Weapon: {t_weapon:.0f}ms"
        cv2.putText(annotated, timing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if out_video is not None:
            out_video.write(annotated)
        
        cv2.imshow('Weapon Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[{frame_count}] FPS: {avg_fps:.1f} | Persons: {len(person_boxes)} | Weapons: {len(weapon_detections)}")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()
    weapon_subprocess.stdin.close()
    weapon_subprocess.wait()
    print(f"\nProcessed {frame_count} frames")
    if args.save:
        print(f"Saved to {args.save}")
