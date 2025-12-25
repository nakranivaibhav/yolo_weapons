import sys
from pathlib import Path
import cv2
import numpy as np
import time
import subprocess
import json
import base64

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEYO_ROOT = PROJECT_ROOT / "DEYO"

import argparse
parser = argparse.ArgumentParser(description='DEYO Person + YOLO Weapon + Classifier Pipeline')
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--deyo_model", type=str, default=str(PROJECT_ROOT / "models" / "deyo" / "deyo-x.pt"))
parser.add_argument("--weapon_model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "weapon_detection_yolo11m_640" / "weights" / "best.pt"))
parser.add_argument("--classifier_model", type=str, default="/workspace/yolo_dataset_cls_5fold/predictions/fold_0_model")
parser.add_argument("--person_conf", type=float, default=0.3)
parser.add_argument("--weapon_conf", type=float, default=0.25)
parser.add_argument("--classifier_conf", type=float, default=0.5, help="Classifier confidence threshold")
parser.add_argument("--iou", type=float, default=0.45)
parser.add_argument("--roi_expand", type=float, default=0.15)
parser.add_argument("--crop_expand", type=float, default=0.1, help="Expansion for YOLO detection crops sent to classifier")
parser.add_argument("--downscale", type=float, default=0.5)
parser.add_argument("--max_frames", type=int, default=60)
parser.add_argument("--track", action="store_true", help="Enable ByteTrack for temporal filtering")
parser.add_argument("--track_persist", type=int, default=30, help="Frames to persist track after disappearance")
parser.add_argument("--min_hits", type=int, default=3, help="Minimum hits before track is confirmed")
parser.add_argument("--use_classifier_class", action="store_true", help="Use classifier class instead of YOLO class")
parser.add_argument("--filter_human", action="store_true", help="Filter out detections classified as 'human'")
parser.add_argument("--filter_cellphone", action="store_true", help="Filter out detections classified as 'cell_phone'")
parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "inference_output" / "person_weapon_classifier.mp4"))
args = parser.parse_args()

WEAPON_LABELS_YOLO = {0: 'knife', 1: 'gun', 2: 'rifle', 3: 'baseball_bat'}
CLASSIFIER_LABELS = {0: 'baseball_bat', 1: 'cell_phone', 2: 'gun', 3: 'human', 4: 'knife', 5: 'rifle'}
CLASSIFIER_LABEL_TO_ID = {v: k for k, v in CLASSIFIER_LABELS.items()}

DANGEROUS_CLASSES = {'gun', 'rifle', 'knife', 'baseball_bat'}

print(f"\n{'='*80}")
print("PERSON + WEAPON + CLASSIFIER PIPELINE")
print(f"{'='*80}\n")

print("[1/3] Loading DEYO person model...")
sys.path.insert(0, str(DEYO_ROOT))
from ultralytics import RTDETR
person_model = RTDETR(args.deyo_model)
print("      ✓ DEYO loaded")

print("[2/3] Loading YOLO weapon model (subprocess)...")
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
print("      ✓ YOLO weapon model loaded")

print("[3/3] Loading Classifier model (subprocess)...")
classifier_subprocess = subprocess.Popen(
    [sys.executable, str(SCRIPT_DIR / "classifier_subprocess.py"), 
     args.classifier_model, str(args.classifier_conf)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

while True:
    line = classifier_subprocess.stderr.readline()
    if "CLASSIFIER_MODEL_READY" in line:
        break
    if not line:
        raise RuntimeError("Classifier subprocess failed to start")
print("      ✓ ConvNextV2 classifier loaded\n")

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
print(f"Weapon detection: Full-res crops")
print(f"Classifier: YOLO detection crops\n")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out_video = cv2.VideoWriter(args.out, fourcc, fps, (width_person, height_person))

stats = {
    'person_count': [],
    'weapon_count': [],
    'classifier_count': [],
    'filtered_count': [],
    'person_times': [],
    'weapon_times': [],
    'classifier_times': []
}

detection_log = []

print(f"Processing {args.max_frames} frames...\n")

def expand_box(x1, y1, x2, y2, expand_ratio, max_w, max_h):
    w = x2 - x1
    h = y2 - y1
    exp_w = int(w * expand_ratio)
    exp_h = int(h * expand_ratio)
    return (
        max(0, int(x1 - exp_w)),
        max(0, int(y1 - exp_h)),
        min(max_w, int(x2 + exp_w)),
        min(max_h, int(y2 + exp_h))
    )

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
    classifier_results = []
    t_weapon = 0
    t_classifier = 0
    
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
                
                weapon_detections.append({
                    'bbox_orig': [x1_global, y1_global, x2_global, y2_global],
                    'yolo_conf': conf_w,
                    'yolo_cls': int(cls_w),
                    'yolo_name': WEAPON_LABELS_YOLO[int(cls_w)],
                    'person': [x1_p, y1_p, x2_p, y2_p, conf_p]
                })
        
        t_weapon = (time.perf_counter() - t_start) * 1000
        
        if len(weapon_detections) > 0:
            t_start = time.perf_counter()
            
            for det in weapon_detections:
                x1_g, y1_g, x2_g, y2_g = det['bbox_orig']
                
                x1_crop, y1_crop, x2_crop, y2_crop = expand_box(
                    x1_g, y1_g, x2_g, y2_g, 
                    args.crop_expand, width_orig, height_orig
                )
                
                weapon_crop = frame_orig[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if weapon_crop.size == 0:
                    det['cls_result'] = None
                    continue
                
                crop_bytes = weapon_crop.tobytes()
                crop_b64 = base64.b64encode(crop_bytes).decode('utf-8')
                
                request = json.dumps({
                    'crop': crop_b64,
                    'shape': weapon_crop.shape
                })
                
                classifier_subprocess.stdin.write(request + '\n')
                classifier_subprocess.stdin.flush()
                
                response = classifier_subprocess.stdout.readline()
                cls_result = json.loads(response.strip())
                det['cls_result'] = cls_result
                classifier_results.append(cls_result)
            
            t_classifier = (time.perf_counter() - t_start) * 1000
    
    filtered_detections = []
    for det in weapon_detections:
        cls_result = det.get('cls_result')
        if cls_result is None:
            continue
        
        if args.filter_human and cls_result['class_name'] == 'human':
            continue
        if args.filter_cellphone and cls_result['class_name'] == 'cell_phone':
            continue
        
        if cls_result['confidence'] < args.classifier_conf:
            continue
        
        if args.use_classifier_class:
            final_class_name = cls_result['class_name']
            final_class_id = cls_result['class']
            final_conf = cls_result['confidence']
        else:
            final_class_name = det['yolo_name']
            final_class_id = det['yolo_cls']
            final_conf = det['yolo_conf']
        
        x1_scaled = det['bbox_orig'][0] / scale_x
        y1_scaled = det['bbox_orig'][1] / scale_y
        x2_scaled = det['bbox_orig'][2] / scale_x
        y2_scaled = det['bbox_orig'][3] / scale_y
        
        filtered_detections.append({
            'weapon': [x1_scaled, y1_scaled, x2_scaled, y2_scaled, final_conf, final_class_id],
            'class_name': final_class_name,
            'yolo_name': det['yolo_name'],
            'classifier_name': cls_result['class_name'],
            'classifier_conf': cls_result['confidence'],
            'yolo_conf': det['yolo_conf'],
            'person': det['person']
        })
    
    if args.track and tracker is not None and len(filtered_detections) > 0:
        dets_np = np.array([[d['weapon'][0] * scale_x, d['weapon'][1] * scale_y, 
                             d['weapon'][2] * scale_x, d['weapon'][3] * scale_y, 
                             d['weapon'][4], d['weapon'][5]] for d in filtered_detections])
        
        tracks = tracker.update(dets_np, frame_orig)
        
        tracked_detections = []
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                
                best_match = None
                best_iou = 0
                for fd in filtered_detections:
                    fx1, fy1, fx2, fy2 = [c * scale_x if i % 2 == 0 else c * scale_y 
                                          for i, c in enumerate(fd['weapon'][:4])]
                    
                    xi1, yi1 = max(x1, fx1), max(y1, fy1)
                    xi2, yi2 = min(x2, fx2), min(y2, fy2)
                    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
                    union = (x2-x1)*(y2-y1) + (fx2-fx1)*(fy2-fy1) - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = fd
                
                tracked_det = {
                    'weapon': [x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y, conf, int(cls)],
                    'track_id': int(track_id),
                    'class_name': best_match['class_name'] if best_match else WEAPON_LABELS_YOLO.get(int(cls), 'unknown'),
                    'classifier_name': best_match['classifier_name'] if best_match else None,
                    'classifier_conf': best_match['classifier_conf'] if best_match else None
                }
                tracked_detections.append(tracked_det)
        filtered_detections = tracked_detections
    elif args.track and tracker is not None:
        tracker.update(np.empty((0, 6)), frame_orig)
    
    stats['person_count'].append(len(person_boxes))
    stats['weapon_count'].append(len(weapon_detections))
    stats['classifier_count'].append(len(classifier_results))
    stats['filtered_count'].append(len(filtered_detections))
    stats['person_times'].append(t_person)
    stats['weapon_times'].append(t_weapon)
    stats['classifier_times'].append(t_classifier)
    
    for det in filtered_detections:
        detection_log.append({
            'frame': frame_idx,
            'detection': det
        })
    
    annotated = frame_person.copy()
    
    for det in filtered_detections:
        x1_w, y1_w, x2_w, y2_w, conf_w, cls_w = det['weapon']
        class_name = det['class_name']
        
        if class_name in ['gun', 'rifle']:
            color = (0, 0, 255)
        elif class_name in ['knife', 'baseball_bat']:
            color = (0, 165, 255)
        elif class_name == 'cell_phone':
            color = (255, 165, 0)
        elif class_name == 'human':
            color = (128, 128, 128)
        else:
            color = (0, 255, 255)
        
        cv2.rectangle(annotated, (int(x1_w), int(y1_w)), (int(x2_w), int(y2_w)), color, 3)
        
        if args.track and 'track_id' in det:
            label = f"ID:{det['track_id']} {class_name} {conf_w:.2f}"
        else:
            if det.get('classifier_conf'):
                label = f"{class_name} (Y:{det.get('yolo_conf', conf_w):.2f}/C:{det['classifier_conf']:.2f})"
            else:
                label = f"{class_name} {conf_w:.2f}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (int(x1_w), int(y1_w)-th-10), (int(x1_w)+tw, int(y1_w)), color, -1)
        cv2.putText(annotated, label, (int(x1_w), int(y1_w)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    out_video.write(annotated)
    
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}: {len(person_boxes)} persons, {len(weapon_detections)} YOLO dets, "
              f"{len(filtered_detections)} filtered | "
              f"P:{t_person:.0f}ms W:{t_weapon:.0f}ms C:{t_classifier:.0f}ms")

cap.release()
out_video.release()

weapon_subprocess.stdin.close()
weapon_subprocess.wait()
classifier_subprocess.stdin.close()
classifier_subprocess.wait()

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"Frames processed: {len(stats['person_count'])}")
print(f"Total persons detected: {sum(stats['person_count'])}")
print(f"Total YOLO weapon detections: {sum(stats['weapon_count'])}")
print(f"Total after classifier filter: {sum(stats['filtered_count'])}")
print(f"Avg person detection time: {np.mean(stats['person_times']):.1f}ms")
if any(t > 0 for t in stats['weapon_times']):
    print(f"Avg weapon detection time: {np.mean([t for t in stats['weapon_times'] if t > 0]):.1f}ms")
if any(t > 0 for t in stats['classifier_times']):
    print(f"Avg classifier time: {np.mean([t for t in stats['classifier_times'] if t > 0]):.1f}ms")

log_path = Path(args.out).with_suffix('.json')
with open(log_path, 'w') as f:
    json.dump({
        'args': vars(args),
        'stats': stats,
        'detections': detection_log
    }, f, indent=2, default=str)
print(f"\n📊 Detection log: {log_path}")
print(f"📹 Output video: {args.out}\n")

