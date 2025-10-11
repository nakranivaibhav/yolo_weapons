#!/workspace/yolo_train/.venv/bin/python

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

parser = argparse.ArgumentParser(description='Manual Tiled Inference with TensorRT for real-time 4K detection')
parser.add_argument("--video", type=str, default="/workspace/yolo_infer/john_wick_end.mkv")
parser.add_argument("--model", type=str, default="/workspace/yolo_train/weapon_detection/weapon_detection_yolo11s_640/weights/best_int8.engine")
parser.add_argument("--tile_size", type=int, default=640, help="Size of each tile")
parser.add_argument("--overlap", type=int, default=128, help="Overlap pixels between tiles")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
parser.add_argument("--camera_fps", type=int, default=30)
parser.add_argument("--max_frames", type=int, default=600)
parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor (0.5 = half resolution)")
parser.add_argument("--batch_tiles", action="store_true", help="Batch process tiles")
parser.add_argument("--refine_rois", action="store_true", help="Refine detections with full-res crops")
parser.add_argument("--roi_expand", type=float, default=0.2, help="Expand ROI by this fraction for refinement")
parser.add_argument("--refine_conf", type=float, default=0.30, help="Confidence threshold for refined ROIs")
parser.add_argument("--track", action="store_true", help="Enable ByteTrack tracking")
parser.add_argument("--track_persist", type=int, default=30, help="Frames to persist track after disappearance")
parser.add_argument("--min_hits", type=int, default=3, help="Minimum hits before track is confirmed (higher = fewer FPs)")
parser.add_argument("--save_vis", action="store_true")
parser.add_argument("--out", type=str, default="./tiled_out")
args = parser.parse_args()

print(f"\n{'='*80}")
print(f"TILED TENSORRT INFERENCE - YOLO-s")
print(f"{'='*80}")
print(f"Model: {Path(args.model).name}")
print(f"Tile size: {args.tile_size}x{args.tile_size}")
print(f"Overlap: {args.overlap}px")
print(f"Batched: {args.batch_tiles}")
print(f"Target: {args.camera_fps} FPS real-time\n")

print(f"[1/2] Loading YOLO-s TensorRT model...")
model = YOLO(args.model, task='detect')

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(width_orig * args.downscale)
height = int(height_orig * args.downscale)
print(f"[2/2] Video: {width_orig}x{height_orig}")
if args.downscale != 1.0:
    print(f"      Scaled to: {width}x{height} ({args.downscale*100:.0f}%)")

def create_tiles(img_w, img_h, tile_size, overlap):
    stride = tile_size - overlap
    tiles = []
    
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            
            if (x2 - x1) < tile_size // 2 or (y2 - y1) < tile_size // 2:
                continue
            
            tiles.append((x1, y1, x2, y2))
    
    return tiles

tile_coords = create_tiles(width, height, args.tile_size, args.overlap)
print(f"Tiles per frame: {len(tile_coords)}")

print(f"\n[INFO] Warming up TensorRT...")
dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
dummy_tiles = [dummy_frame[y1:y2, x1:x2] for x1, y1, x2, y2 in tile_coords[:4]]

if args.batch_tiles:
    _ = model(dummy_tiles, imgsz=args.tile_size, conf=args.conf, verbose=False)
else:
    for tile in dummy_tiles:
        _ = model(tile, imgsz=args.tile_size, conf=args.conf, verbose=False)

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
    'refined_detections': [],
    'latencies': [],
    'inference_times': [],
    'refine_times': []
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
        
        if args.batch_tiles:
            try:
                all_results = model(
                    tiles,
                    imgsz=args.tile_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=100,
                    verbose=False,
                    half=True,
                    stream=False
                )
            except Exception as e:
                print(f"Batch inference failed: {e}, falling back to sequential")
                all_results = []
                for tile in tiles:
                    result = model(tile, imgsz=args.tile_size, conf=args.conf, iou=args.iou, verbose=False, half=True)[0]
                    all_results.append(result)
        else:
            all_results = []
            for tile in tiles:
                result = model(
                    tile,
                    imgsz=args.tile_size,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=100,
                    verbose=False,
                    half=True
                )[0]
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
        
        if args.refine_rois and len(merged_dets) > 0 and args.downscale != 1.0:
            t_refine_start = time.perf_counter()
            refined_dets = []
            roi_crops = []
            roi_info = []
            
            for det in merged_dets:
                x1, y1, x2, y2, conf, cls = det
                
                x1_full = int(x1 * scale_x)
                y1_full = int(y1 * scale_y)
                x2_full = int(x2 * scale_x)
                y2_full = int(y2 * scale_y)
                
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
                MAX_BATCH = 4
                num_batches = (len(roi_crops) + MAX_BATCH - 1) // MAX_BATCH
                
                try:
                    all_roi_results = []
                    for i in range(0, len(roi_crops), MAX_BATCH):
                        batch = roi_crops[i:i+MAX_BATCH]
                        
                        if len(batch) < MAX_BATCH:
                            batch = batch + [batch[-1]] * (MAX_BATCH - len(batch))
                        
                        batch_results = model(
                            batch,
                            imgsz=args.tile_size,
                            conf=args.refine_conf,
                            iou=args.iou,
                            max_det=10,
                            verbose=False,
                            half=True,
                            stream=False
                        )
                        
                        actual_count = min(MAX_BATCH, len(roi_crops) - i)
                        all_roi_results.extend(batch_results[:actual_count])
                    
                    verified_count = 0
                    rejected_count = 0
                    
                    for i, result in enumerate(all_roi_results):
                        rx1, ry1, rx2, ry2, orig_conf, orig_cls = roi_info[i]
                        
                        if len(result.boxes) > 0:
                            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
                            bx1, by1, bx2, by2 = best_box.xyxy[0].cpu().numpy()
                            new_conf = float(best_box.conf[0])
                            new_cls = int(best_box.cls[0])
                            
                            final_x1 = int(rx1 + bx1)
                            final_y1 = int(ry1 + by1)
                            final_x2 = int(rx1 + bx2)
                            final_y2 = int(ry1 + by2)
                            
                            refined_dets.append([
                                final_x1 / scale_x,
                                final_y1 / scale_y,
                                final_x2 / scale_x,
                                final_y2 / scale_y,
                                new_conf,
                                new_cls
                            ])
                            verified_count += 1
                        else:
                            rejected_count += 1
                except Exception as e:
                    print(f"ROI refinement failed: {e}, using original detections")
                    refined_dets = merged_dets
            else:
                refined_dets = merged_dets
            
            t_refine = (time.perf_counter() - t_refine_start) * 1000
            final_dets = refined_dets
            refined_count = len(refined_dets)
        else:
            final_dets = merged_dets
            refined_count = merged_det_count
            t_refine = 0.0
        
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
            stats['refined_detections'].append(refined_count)
            stats['latencies'].append(latency)
            stats['inference_times'].append(t_inf)
            stats['refine_times'].append(t_refine)
        
        if args.save_vis:
            annotated = frame_orig.copy()
            for det in final_dets:
                if args.track and len(det) == 7:
                    x1, y1, x2, y2, conf, cls, track_id = det
                else:
                    x1, y1, x2, y2, conf, cls = det
                    track_id = None
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                cls_name = model.names[int(cls)]
                
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
    out_video = cv2.VideoWriter(f"{args.out}/tiled_output.mp4", fourcc, args.camera_fps, (width_orig, height_orig))
    
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
print(f"DETECTION PIPELINE:")
print(f"  1. Tiled Inference    : conf={args.conf} on {width}x{height} ({len(tile_coords)} tiles)")
print(f"  2. SAHI NMS Merge     : iou={args.iou}")
if args.refine_rois:
    print(f"  3. ROI Refinement     : conf={args.refine_conf} on {width_orig}x{height_orig} crops (expand={args.roi_expand})")
else:
    print(f"  3. ROI Refinement     : DISABLED")
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
print(f"TILED TENSORRT REAL-TIME RESULTS")
print(f"{'='*80}")
print(f"\nConfiguration:")
print(f"  Camera FPS: {args.camera_fps}")
print(f"  Tile size: {args.tile_size}x{args.tile_size}")
print(f"  Overlap: {args.overlap}px")
print(f"  Tiles per frame: {len(tile_coords)}")
print(f"  Batched: {args.batch_tiles}")
print(f"  Confidence: {args.conf}")
print(f"\nTimeline:")
print(f"  Total time: {t_total:.2f}s")
print(f"  Frames produced: {stats['frames_produced']}")
print(f"  Frames processed: {stats['frames_processed']}")
print(f"  Dropped: {stats['dropped_frames']} ({stats['dropped_frames']/max(stats['frames_produced'],1)*100:.1f}%)")
print(f"\nDetection:")
print(f"  Avg raw detections: {np.mean(stats['raw_detections']):.2f}")
print(f"  Avg merged detections: {np.mean(stats['merged_detections']):.2f}")
if args.refine_rois:
    print(f"  Avg refined detections: {np.mean(stats['refined_detections']):.2f}")
    refine_reduction = (1 - np.mean(stats['refined_detections'])/max(np.mean(stats['merged_detections']),1))*100
    print(f"  ROI refinement reduction: {refine_reduction:.1f}%")
print(f"  NMS reduction: {(1 - np.mean(stats['merged_detections'])/max(np.mean(stats['raw_detections']),1))*100:.1f}%")
print(f"  Avg tiled inference: {np.mean(stats['inference_times']):.1f}ms")
print(f"  P95 tiled inference: {np.percentile(stats['inference_times'], 95):.1f}ms")
if args.refine_rois and any(t > 0 for t in stats['refine_times']):
    print(f"  Avg ROI refinement: {np.mean([t for t in stats['refine_times'] if t > 0]):.1f}ms")
    print(f"  P95 ROI refinement: {np.percentile([t for t in stats['refine_times'] if t > 0], 95):.1f}ms")
    total_inf = np.mean(stats['inference_times']) + np.mean([t for t in stats['refine_times'] if t > 0])
    print(f"  Total inference: {total_inf:.1f}ms")
print(f"\nEnd-to-end latency:")
print(f"  Average: {np.mean(stats['latencies']):.1f}ms")
print(f"  P95: {np.percentile(stats['latencies'], 95):.1f}ms")
print(f"  P99: {np.percentile(stats['latencies'], 99):.1f}ms")
print(f"  Max: {np.max(stats['latencies']):.1f}ms")
print(f"\nReal-time verdict:")
target_latency = 1000 / args.camera_fps * 2
p95_latency = np.percentile(stats['latencies'], 95)
print(f"  Latency < {target_latency:.0f}ms: {'✅ PASS' if p95_latency < target_latency else '❌ FAIL'} (P95={p95_latency:.1f}ms)")
print(f"  Dropped < 1%: {'✅ PASS' if stats['dropped_frames']/max(stats['frames_produced'],1) < 0.01 else '❌ FAIL'}")
print(f"  Overall: {'✅ REAL-TIME CAPABLE @ ' + str(args.camera_fps) + ' FPS' if stats['dropped_frames'] == 0 and p95_latency < target_latency else '⚠️  MARGINAL' if stats['dropped_frames'] < 5 else '❌ CANNOT KEEP UP'}")

if args.save_vis:
    print(f"\n📹 Output: {args.out}/tiled_output.mp4")

print()

