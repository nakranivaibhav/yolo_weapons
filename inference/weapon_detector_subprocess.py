#!/usr/bin/env python3
import sys
import json
import base64
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

model_path = sys.argv[1]
conf_threshold = float(sys.argv[2])

from ultralytics import YOLO
weapon_model = YOLO(model_path, task='detect')

sys.stderr.write("WEAPON_MODEL_READY\n")
sys.stderr.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    
    data = json.loads(line.strip())
    
    crop_bytes = base64.b64decode(data['crop'])
    crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(data['shape'])
    imgsz = data['imgsz']
    
    results = weapon_model(crop_arr, imgsz=imgsz, conf=conf_threshold, verbose=False, half=True)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append([x1, y1, x2, y2, conf, cls])
    
    print(json.dumps(detections), flush=True)

