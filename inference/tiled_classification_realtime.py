#!/usr/bin/env python3

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from threading import Thread, Lock
from queue import Queue, Empty
from ultralytics import YOLO
import torch
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMSPostprocess
from boxmot import ByteTrack
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
import torch_tensorrt

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

parser = argparse.ArgumentParser(description='Tiled Detection + ConvNeXT Classification for real-time 4K weapon detection')
parser.add_argument("--video", type=str, default=str(PROJECT_ROOT / "data" / "test_video.mp4"))
parser.add_argument("--detect_model", type=str, default=str(PROJECT_ROOT / "models" / "yolo" / "weapon_detection_yolo11s_640" / "weights" / "best_int8.engine"))
parser.add_argument("--classify_model", type=str, default=str(PROJECT_ROOT / "models" / "convnext_compiled" / "convnext_bs4.ep"))
parser.add_argument("--tile_size", type=int, default=1280, help="Size of each tile")
parser.add_argument("--detect_batch", type=int, default=8, help="Batch size for detection model")
parser.add_argument("--classify_batch", type=int, default=4, help="Batch size for classification model")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
parser.add_argument("--camera_fps", type=int, default=30)
parser.add_argument("--max_frames", type=int, default=600)
parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor (0.5 = half resolution)")
parser.add_argument("--batch_tiles", action="store_true", help="Batch process tiles")
parser.add_argument("--classify_rois", action="store_true", help="Classify detections with ConvNeXT")
parser.add_argument("--roi_expand", type=float, default=0.2, help="Expand ROI by this fraction for classification")
parser.add_argument("--classify_conf", type=float, default=0.50, help="Minimum confidence to keep classification")
parser.add_argument("--track", action="store_true", help="Enable ByteTrack tracking")
parser.add_argument("--track_persist", type=int, default=30, help="Frames to persist track after disappearance")
parser.add_argument("--min_hits", type=int, default=3, help="Minimum hits before track is confirmed")
parser.add_argument("--save_vis", action="store_true")
parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "inference_output"))
args = parser.parse_args()

print(f"\n{'='*80}")
print(f"TILED DETECTION + CONVNEXT CLASSIFICATION PIPELINE")
print(f"{'='*80}")
print(f"Detection model: {Path(args.detect_model).name}")
print(f"  - Tile size: {args.tile_size}x{args.tile_size}")
print(f"  - Batch size: {args.detect_batch}")
print(f"  - Downscale: {args.downscale}x ({args.downscale*100:.0f}%)")
print(f"Classification model: {Path(args.classify_model).name}")
print(f"  - Batch size: {args.classify_batch}")
print(f"  - Full-res crops: Yes")
print(f"Target: {args.camera_fps} FPS real-time\n")

print(f"[1/3] Loading YOLO detection model...")
detect_model = YOLO(args.detect_model, task='detect')
YOLO_LABELS = {0: 'knife', 1: 'gun'}

print(f"[2/3] Loading ConvNeXT classification model...")
model_path = Path(args.classify_model)
model_ext = model_path.suffix

if model_ext == '.ts':
    print(f"      Loading TorchScript format (TensorRT-optimized)...")
    classify_model = torch.jit.load(str(model_path)).cuda()
    print(f"      ✓ TorchScript model loaded")
elif model_ext == '.pt2':
    print(f"      Loading PT2 (AOT Inductor) format...")
    classify_model = torch._inductor.aoti_load_package(str(model_path))
    print(f"      ✓ PT2 model loaded (TensorRT-optimized)")
elif model_ext == '.ep':
    print(f"      Loading ExportedProgram format...")
    classify_model = torch.export.load(str(model_path)).module()
elif model_ext == '.pt':
    print(f"      Loading Torch-TensorRT compiled model...")
    try:
        classify_model = torch_tensorrt.load(str(model_path)).module()
        classify_model.eval()
        print(f"      ✓ Torch-TensorRT model loaded")
    except Exception as e:
        print(f"      Torch-TensorRT load failed: {e}")
        print(f"      Trying standard PyTorch load...")
        classify_model = torch.load(str(model_path), weights_only=False)
        classify_model.eval()
        classify_model.cuda()
        print(f"      ✓ Standard PyTorch model loaded")
else:
    raise ValueError(f"Unsupported model format: {model_ext}. Use .pt2, .ep, .ts, or .pt")

if hasattr(classify_model, 'eval'):
    classify_model.eval()
if hasattr(classify_model, 'cuda') and model_ext not in ['.ts', '.pt2']:
    classify_model.cuda()

CONVNEXT_LABELS = {0: 'gun', 1: 'knife'}
YOLO_TO_CONVNEXT = {'knife': 1, 'gun': 0}
CONVNEXT_TO_YOLO = {0: 1, 1: 0}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
classify_transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=mean, std=std)
])

print(f"Label mapping:")
print(f"  YOLO: {YOLO_LABELS}")
print(f"  ConvNeXT: {CONVNEXT_LABELS}")
print(f"  Conversion: YOLO->ConvNeXT = {YOLO_TO_CONVNEXT}")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(width_orig * args.downscale)
height = int(height_orig * args.downscale)
print(f"[3/3] Video: {width_orig}x{height_orig}")
if args.downscale != 1.0:
    print(f"      Scaled to: {width}x{height} ({args.downscale*100:.0f}%)")
    print(f"\n      Coordinate transformation:")
    print(f"        Detection runs on : {width}x{height} (downscaled)")
    print(f"        Classification on : {width_orig}x{height_orig} (full-res crops)")
    print(f"        Scale factors     : {width_orig/width:.1f}x, {height_orig/height:.1f}x")

def create_tiles(img_w, img_h, tile_size, overlap=None):
    if img_w == 1920 and img_h == 804 and tile_size == 640:
        tiles = [
            (0, 0, 640, 640),
            (427, 0, 1067, 640),
            (853, 0, 1493, 640),
            (1280, 0, 1920, 640),
            (0, 164, 640, 804),
            (427, 164, 1067, 804),
            (853, 164, 1493, 804),
            (1280, 164, 1920, 804),
        ]
        
        print(f"Tiling strategy (hardcoded for 1920x804 @ 0.5x downscale):")
        print(f"  Width {img_w}px: 4 tiles, overlap=213px (33.3%)")
        print(f"  Height {img_h}px: 2 tiles, overlap=476px (74.4%)")
        print(f"  Total tiles: 8 (perfect for batch_size=8)")
        
        return tiles
    else:
        raise ValueError(f"Unsupported dimensions: {img_w}x{img_h}. Expected 1920x804 (downscaled 0.5x from 3840x1608).")

tile_coords = create_tiles(width, height, args.tile_size)
print(f"Total tiles per frame: {len(tile_coords)}")

if args.downscale == 0.5:
    assert width == 1920 and height == 804, f"Expected 1920x804 for 0.5x downscale, got {width}x{height}"
    assert args.tile_size == 640, f"Expected tile_size=640 for downscaled mode, got {args.tile_size}"
    print(f"[VALIDATION] Downscale configuration verified: {width}x{height} with {args.tile_size}x{args.tile_size} tiles")

print(f"\n[INFO] Warming up models...")
DETECTION_BATCH_SIZE = args.detect_batch
dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
dummy_tiles = [dummy_frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tile_coords[:DETECTION_BATCH_SIZE]]

print(f"[INFO] Warming up detection model (batch_size={DETECTION_BATCH_SIZE}, tile_size={args.tile_size})...")
_ = detect_model(dummy_tiles, imgsz=args.tile_size, conf=args.conf, verbose=False, half=True)

if args.classify_rois:
    print(f"[INFO] Warming up classification model (batch_size={args.classify_batch})...")
    dummy_crops = torch.randn(args.classify_batch, 3, 224, 224).cuda()
    with torch.no_grad():
        _ = classify_model(dummy_crops)

print(f"[INFO] Warmup complete\n")

if args.track:
    import logging
    logging.getLogger('boxmot').setLevel(logging.WARNING)
    
    tracker = ByteTrack(
        track_thresh=args.conf,
        track_buffer=args.track_persist,
        match_thresh=0.8,
        frame_rate=args.camera_fps,
        min_hits=args.min_hits
    )
    print(f"[INFO] ByteTrack initialized (persist={args.track_persist} frames, min_hits={args.min_hits})\n")
else:
    tracker = None

frame_interval = 1.0 / args.camera_fps
frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=100) if args.save_vis else None
stop_flag = False

stats = {
    'frames_produced': 0,
    'frames_processed': 0,
    'dropped_frames': 0,
    'raw_detections': [],
    'merged_detections': [],
    'classified_detections': [],
    'rejected_by_classification': [],
    'latencies': [],
    'inference_times': [],
    'classify_times': [],
    'classification_confidences': []
}
stats_lock = Lock()

def nms_merge_tiles_sahi(all_dets, iou_thresh=0.45):
    if len(all_dets) == 0:
        return []
    
    object_predictions = []
    for det in all_dets:
        x1, y1, x2, y2, conf, cls = det
        
        obj_pred = ObjectPrediction(
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            category_id=int(cls),
            category_name=str(int(cls)),
            score=float(conf),
            shift_amount=[0, 0],
            full_shape=None
        )
        object_predictions.append(obj_pred)
    
    nms = NMSPostprocess(match_threshold=iou_thresh, match_metric="IOU", class_agnostic=True)
    merged_predictions = nms(object_predictions)
    
    merged_dets = []
    for pred in merged_predictions:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        merged_dets.append([x1, y1, x2, y2, pred.score.value, pred.category.id])
    
    return merged_dets

def classify_rois_batch(roi_crops, roi_info, frame_orig, scale_x, scale_y, classify_batch_size=4):
    if len(roi_crops) == 0:
        return [], 0, []
    
    BATCH_SIZE = classify_batch_size
    classified_dets = []
    rejected_count = 0
    all_confidences = []
    
    preprocessed = []
    for crop in roi_crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        from PIL import Image
        crop_pil = Image.fromarray(crop_rgb)
        tensor = classify_transform(crop_pil)
        preprocessed.append(tensor)
    
    for i in range(0, len(preprocessed), BATCH_SIZE):
        batch_tensors = preprocessed[i:i+BATCH_SIZE]
        batch_info = roi_info[i:i+BATCH_SIZE]
        
        if len(batch_tensors) < BATCH_SIZE:
            padding_needed = BATCH_SIZE - len(batch_tensors)
            batch_tensors = batch_tensors + [batch_tensors[-1]] * padding_needed
        
        batch_input = torch.stack(batch_tensors).cuda()
        
        with torch.no_grad():
            outputs = classify_model(batch_input)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs.logits
        
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = logits.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
        
        actual_count = len(batch_info)
        for j in range(actual_count):
            rx1, ry1, rx2, ry2, orig_conf, orig_yolo_cls = batch_info[j]
            
            convnext_cls = predicted_classes[j].item()
            confidence = confidences[j].item()
            all_confidences.append(confidence)
            
            yolo_cls = CONVNEXT_TO_YOLO[convnext_cls]
            
            if confidence >= args.classify_conf:
                classified_dets.append([
                    rx1 / scale_x,
                    ry1 / scale_y,
                    rx2 / scale_x,
                    ry2 / scale_y,
                    confidence,
                    yolo_cls
                ])
            else:
                rejected_count += 1
    
    return classified_dets, rejected_count, all_confidences

def frame_producer():
    frame_idx = 0
    next_frame_time = time.perf_counter()
    
    while frame_idx < args.max_frames:
        current_time = time.perf_counter()
        
        if current_time >= next_frame_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                frame_queue.put((frame.copy(), current_time, frame_idx), block=False)
                with stats_lock:
                    stats['frames_produced'] = frame_idx + 1
            except:
                with stats_lock:
                    stats['dropped_frames'] += 1
                print(f"⚠️  Frame {frame_idx} dropped - backlog!")
            
            frame_idx += 1
            next_frame_time += frame_interval
        else:
            time.sleep(0.0005)
    
    global stop_flag
    stop_flag = True
    cap.release()

def tiled_worker():
    """
    COORDINATE SPACE SUMMARY:
    - frame_orig: FULL-RES (3840x1608)
    - frame: DOWNSCALED (1920x804 when downscale=0.5)
    - scale_x, scale_y: Conversion factors (2.0 when downscale=0.5)
    
    FLOW:
    1. Detection: runs on DOWNSCALED frame → outputs in DOWNSCALED coords
    2. Classification: crops from FULL-RES frame_orig (scale UP, then back DOWN)
    3. Tracking: runs on FULL-RES (scale UP, then back DOWN)
    4. Visualization: draws on DOWNSCALED frame (no scaling needed)
    """
    while not stop_flag or not frame_queue.empty():
        try:
            frame_orig, t_arrive, frame_idx = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        
        if args.downscale != 1.0:
            frame = cv2.resize(frame_orig, (width, height), interpolation=cv2.INTER_LINEAR)
            scale_x = width_orig / width
            scale_y = height_orig / height
        else:
            frame = frame_orig
            scale_x = 1.0
            scale_y = 1.0
        
        tiles = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tile_coords]
        
        t_inf_start = time.perf_counter()
        
        DETECTION_BATCH_SIZE = args.detect_batch
        all_results = []
        
        for i in range(0, len(tiles), DETECTION_BATCH_SIZE):
            batch_tiles = tiles[i:i+DETECTION_BATCH_SIZE]
            
            if len(batch_tiles) < DETECTION_BATCH_SIZE:
                padding_needed = DETECTION_BATCH_SIZE - len(batch_tiles)
                batch_tiles = batch_tiles + [batch_tiles[-1]] * padding_needed
            
            try:
                batch_results = detect_model(
                    batch_tiles,
                    imgsz=args.tile_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=100,
                    verbose=False,
                    half=True,
                    stream=False
                )
                
                actual_count = min(DETECTION_BATCH_SIZE, len(tiles) - i)
                all_results.extend(list(batch_results)[:actual_count])
                
            except Exception as e:
                print(f"Batch detection failed: {e}, processing individually")
                for tile in batch_tiles[:actual_count]:
                    result = detect_model(tile, imgsz=args.tile_size, conf=args.conf, iou=args.iou, verbose=False, half=True)[0]
                    all_results.append(result)
        
        t_inf = (time.perf_counter() - t_inf_start) * 1000
        
        all_dets = []
        for idx, result in enumerate(all_results):
            x1_tile, y1_tile, x2_tile, y2_tile = tile_coords[idx]
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1_global = int(x1 + x1_tile)
                y1_global = int(y1 + y1_tile)
                x2_global = int(x2 + x1_tile)
                y2_global = int(y2 + y1_tile)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                all_dets.append([x1_global, y1_global, x2_global, y2_global, conf, cls])
        
        raw_det_count = len(all_dets)
        merged_dets = nms_merge_tiles_sahi(all_dets, iou_thresh=args.iou)
        merged_det_count = len(merged_dets)
        
        if args.classify_rois and len(merged_dets) > 0:
            t_classify_start = time.perf_counter()
            roi_crops = []
            roi_info = []
            
            for det in merged_dets:
                x1, y1, x2, y2, conf, cls = det
                
                x1_full = int(x1 * scale_x)
                y1_full = int(y1 * scale_y)
                x2_full = int(x2 * scale_x)
                y2_full = int(y2 * scale_y)
                
                if x2_full > width_orig or y2_full > height_orig:
                    print(f"⚠️  Detection coords out of bounds: ({x1_full},{y1_full},{x2_full},{y2_full}) > ({width_orig},{height_orig})")
                    continue
                
                w = x2_full - x1_full
                h = y2_full - y1_full
                expand_w = int(w * args.roi_expand)
                expand_h = int(h * args.roi_expand)
                
                x1_exp = max(0, x1_full - expand_w)
                y1_exp = max(0, y1_full - expand_h)
                x2_exp = min(width_orig, x2_full + expand_w)
                y2_exp = min(height_orig, y2_full + expand_h)
                
                crop = frame_orig[y1_exp:y2_exp, x1_exp:x2_exp]
                if crop.size > 0:
                    roi_crops.append(crop)
                    roi_info.append((x1_exp, y1_exp, x2_exp, y2_exp, conf, cls))
            
            if len(roi_crops) > 0:
                try:
                    classified_dets, rejected_count, confidences = classify_rois_batch(roi_crops, roi_info, frame_orig, scale_x, scale_y, args.classify_batch)
                    final_dets = classified_dets
                    t_classify = (time.perf_counter() - t_classify_start) * 1000
                except Exception as e:
                    print(f"Classification failed: {e}, using original detections")
                    final_dets = merged_dets
                    t_classify = 0.0
                    rejected_count = 0
                    confidences = []
            else:
                final_dets = merged_dets
                t_classify = 0.0
                rejected_count = 0
                confidences = []
            
            classified_count = len(final_dets)
        else:
            final_dets = merged_dets
            classified_count = merged_det_count
            t_classify = 0.0
            rejected_count = 0
            confidences = []
        
        if args.track and tracker is not None:
            if len(final_dets) > 0:
                dets_np = np.array(final_dets)
                dets_np[:, :4] *= [scale_x, scale_y, scale_x, scale_y]
                
                tracks = tracker.update(dets_np, frame_orig)
                
                tracked_dets = []
                if len(tracks) > 0:
                    for track in tracks:
                        x1, y1, x2, y2, track_id, conf, cls = track[:7]
                        tracked_dets.append([
                            x1 / scale_x, y1 / scale_y, 
                            x2 / scale_x, y2 / scale_y, 
                            conf, cls, int(track_id)
                        ])
                
                final_dets = tracked_dets
            else:
                tracker.update(np.empty((0, 6)), frame_orig)
        
        latency = (time.perf_counter() - t_arrive) * 1000
        
        with stats_lock:
            stats['frames_processed'] += 1
            stats['raw_detections'].append(raw_det_count)
            stats['merged_detections'].append(merged_det_count)
            stats['classified_detections'].append(classified_count)
            stats['rejected_by_classification'].append(rejected_count)
            stats['latencies'].append(latency)
            stats['inference_times'].append(t_inf)
            stats['classify_times'].append(t_classify)
            stats['classification_confidences'].extend(confidences)
        
        if args.save_vis:
            annotated = frame.copy()
            for det in final_dets:
                if args.track and len(det) == 7:
                    x1, y1, x2, y2, conf, cls, track_id = det
                else:
                    x1, y1, x2, y2, conf, cls = det
                    track_id = None
                
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cls_name = YOLO_LABELS[int(cls)]
                
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                if track_id is not None:
                    label = f"ID:{track_id} {cls_name} {conf:.2f}"
                else:
                    label = f"{cls_name} {conf:.2f}"
                
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            result_queue.put((annotated, frame_idx))

if args.save_vis:
    import os
    os.makedirs(args.out, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f"{args.out}/classified_output.mp4", fourcc, args.camera_fps, (width, height))
    
    def video_writer():
        while not stop_flag or not result_queue.empty():
            try:
                frame, idx = result_queue.get(timeout=0.1)
                out_video.write(frame)
            except Empty:
                continue
        out_video.release()
    
    writer_thread = Thread(target=video_writer, daemon=True)
    writer_thread.start()

print(f"{'='*80}")
print(f"DETECTION + CLASSIFICATION PIPELINE:")
print(f"  1. Tiled Detection    : YOLO-s {args.tile_size}x{args.tile_size} conf={args.conf}")
print(f"     - Inference on     : {width}x{height} ({args.downscale*100:.0f}% of {width_orig}x{height_orig})")
print(f"     - Tiles per frame  : {len(tile_coords)} (batch_size={args.detect_batch})")
print(f"  2. SAHI NMS Merge     : iou={args.iou}")
if args.classify_rois:
    print(f"  3. ConvNeXT Classify  : conf={args.classify_conf}")
    print(f"     - Crops from       : {width_orig}x{height_orig} (full resolution)")
    print(f"     - ROI expansion    : {args.roi_expand*100:.0f}%")
    print(f"     - Batch size       : {args.classify_batch}")
else:
    print(f"  3. ConvNeXT Classify  : DISABLED")
if args.track:
    print(f"  4. ByteTrack Tracking : min_hits={args.min_hits}, persist={args.track_persist} frames")
else:
    print(f"  4. ByteTrack Tracking : DISABLED")
print(f"{'='*80}")
print(f"Starting @ {args.camera_fps} FPS...")
print(f"Target: {args.max_frames} frames ({args.max_frames/args.camera_fps:.1f}s)\n")

producer_thread = Thread(target=frame_producer, daemon=True)
tiled_thread = Thread(target=tiled_worker, daemon=True)

t_start = time.perf_counter()
producer_thread.start()
tiled_thread.start()

while not stop_flag:
    time.sleep(0.5)

producer_thread.join()
tiled_thread.join()
if args.save_vis:
    writer_thread.join()

t_total = time.perf_counter() - t_start

print(f"\n{'='*80}")
print(f"TILED DETECTION + CLASSIFICATION RESULTS")
print(f"{'='*80}")
print(f"\nConfiguration:")
print(f"  Camera FPS: {args.camera_fps}")
print(f"  Tile size: {args.tile_size}x{args.tile_size}")
print(f"  Tiles per frame: {len(tile_coords)}")
print(f"  Detection batch: {args.detect_batch}")
print(f"  Classification batch: {args.classify_batch}")
print(f"  Detection confidence: {args.conf}")
if args.classify_rois:
    print(f"  Classification confidence: {args.classify_conf}")
print(f"\nTimeline:")
print(f"  Total time: {t_total:.2f}s")
print(f"  Frames produced: {stats['frames_produced']}")
print(f"  Frames processed: {stats['frames_processed']}")
print(f"  Dropped: {stats['dropped_frames']} ({stats['dropped_frames']/max(stats['frames_produced'],1)*100:.1f}%)")
print(f"\nDetection:")
print(f"  Avg raw detections: {np.mean(stats['raw_detections']):.2f}")
print(f"  Avg merged detections: {np.mean(stats['merged_detections']):.2f}")
if args.classify_rois:
    print(f"  Avg classified detections: {np.mean(stats['classified_detections']):.2f}")
    print(f"  Avg rejected by classification: {np.mean(stats['rejected_by_classification']):.2f}")
    classification_keep_rate = np.mean(stats['classified_detections'])/max(np.mean(stats['merged_detections']),1)*100
    print(f"  Classification keep rate: {classification_keep_rate:.1f}%")
    if len(stats['classification_confidences']) > 0:
        print(f"\n  Classification confidence stats:")
        print(f"    Mean: {np.mean(stats['classification_confidences']):.3f}")
        print(f"    Median: {np.median(stats['classification_confidences']):.3f}")
        print(f"    Min: {np.min(stats['classification_confidences']):.3f}")
        print(f"    P5: {np.percentile(stats['classification_confidences'], 5):.3f}")
        print(f"    P95: {np.percentile(stats['classification_confidences'], 95):.3f}")
        print(f"    Max: {np.max(stats['classification_confidences']):.3f}")
print(f"  NMS reduction: {(1 - np.mean(stats['merged_detections'])/max(np.mean(stats['raw_detections']),1))*100:.1f}%")
print(f"  Avg tiled detection: {np.mean(stats['inference_times']):.1f}ms")
print(f"  P95 tiled detection: {np.percentile(stats['inference_times'], 95):.1f}ms")
if args.classify_rois and any(t > 0 for t in stats['classify_times']):
    print(f"  Avg classification: {np.mean([t for t in stats['classify_times'] if t > 0]):.1f}ms")
    print(f"  P95 classification: {np.percentile([t for t in stats['classify_times'] if t > 0], 95):.1f}ms")
    total_inf = np.mean(stats['inference_times']) + np.mean([t for t in stats['classify_times'] if t > 0])
    print(f"  Total inference: {total_inf:.1f}ms")
print(f"\nEnd-to-end latency:")
print(f"  Average: {np.mean(stats['latencies']):.1f}ms")
print(f"  P95: {np.percentile(stats['latencies'], 95):.1f}ms")
print(f"  P99: {np.percentile(stats['latencies'], 99):.1f}ms")
print(f"  Max: {np.max(stats['latencies']):.1f}ms")
print(f"\nReal-time verdict:")
frame_time = 1000 / args.camera_fps
p95_latency = np.percentile(stats['latencies'], 95)
drop_rate = stats['dropped_frames']/max(stats['frames_produced'],1)

if args.classify_rois and any(t > 0 for t in stats['classify_times']):
    avg_inference = np.mean(stats['inference_times']) + np.mean([t for t in stats['classify_times'] if t > 0])
else:
    avg_inference = np.mean(stats['inference_times'])

print(f"  Target frame time: {frame_time:.1f}ms ({args.camera_fps} FPS)")
print(f"  Avg total inference: {avg_inference:.1f}ms")
print(f"  Inference < {frame_time:.1f}ms: {'✅ PASS' if avg_inference < frame_time else '❌ FAIL'}")
print(f"  Dropped frames: {stats['dropped_frames']}/{stats['frames_produced']} ({drop_rate*100:.2f}%)")
print(f"  Dropped < 2%: {'✅ PASS' if drop_rate < 0.02 else '❌ FAIL'}")

if drop_rate < 0.02 and avg_inference < frame_time:
    print(f"  Overall: ✅ REAL-TIME CAPABLE @ {args.camera_fps} FPS")
elif drop_rate < 0.05 and avg_inference < frame_time * 1.5:
    print(f"  Overall: ⚠️  MARGINAL (close to real-time)")
else:
    print(f"  Overall: ❌ CANNOT KEEP UP")

if args.save_vis:
    print(f"\n📹 Output: {args.out}/classified_output.mp4")

print()

