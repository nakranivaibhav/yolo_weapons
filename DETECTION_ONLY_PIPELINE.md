# Detection-Only Pipeline (YOLO TensorRT)
**Tiled Detection + ROI Refinement + ByteTrack**

---

## 📊 Pipeline Overview

### Pipeline Flow (Detection-Only Approach)
```
┌─────────────────┐
│   4K Video      │
│   3840×1608     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Downscale 0.5x │
│   1920×804      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tile into       │
│ overlapping     │
│ 640×640 pieces  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLO Detection  │
│ (all tiles)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SAHI NMS Merge  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ROI Refinement  │
│ (OPTIONAL)      │
│ Re-detect on    │
│ full-res crops  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ByteTrack     │
│ Multi-object    │
│   Tracking      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │
└─────────────────┘
```

---

## 🎯 Key Difference from Two-Stage Pipeline

### Detection-Only vs Two-Stage

**This Pipeline (Detection-Only):**
```
YOLO Detection → SAHI NMS → ROI Refinement (YOLO again) → ByteTrack
```

**Two-Stage Pipeline:**
```
YOLO Detection → SAHI NMS → ConvNeXT Classification → ByteTrack
```

### Advantages
✅ **Simpler** - Single model (YOLO only)
✅ **Faster** - No separate classification model
✅ **Lower memory** - One model loaded
✅ **Good for gun/knife detection** when YOLO is well-trained

### Disadvantages
❌ **Less refinement** - No specialized classifier
❌ **More false positives** - YOLO can confuse similar objects
❌ **Limited to YOLO classes** - Can't add post-detection verification

---

## 1️⃣ Tiled Detection Strategy

### Dynamic Tiling with Overlap

Unlike the hardcoded 8-tile approach, this uses **dynamic tiling** based on overlap parameter:

```python
def create_tiles(img_w, img_h, tile_size, overlap):
    stride = tile_size - overlap
    tiles = []
    
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x1, y1 = x, y
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            
            # Skip tiny edge tiles
            if (x2 - x1) < tile_size // 2 or (y2 - y1) < tile_size // 2:
                continue
            
            tiles.append((x1, y1, x2, y2))
```

### Overlap Calculation

**For 1920×804 with tile_size=640, overlap=128:**

```
stride = 640 - 128 = 512 pixels

Horizontal tiles:
x=0:     0 to 640
x=512:   512 to 1152  (128px overlap with previous)
x=1024:  1024 to 1664 (128px overlap)
x=1536:  1536 to 1920 (partial tile, might be skipped)

Vertical tiles:
y=0:     0 to 640
y=512:   512 to 804   (partial height, might be skipped)

Result: ~6-8 tiles per frame
```

### Overlap Percentage
```
overlap_px = 128
tile_size = 640
overlap_% = (128 / 640) * 100 = 20%
```

**Why 20% overlap?**
- Ensures objects at tile boundaries get detected
- Lower than 33% used in 2-stage pipeline
- Fewer total tiles = faster processing

---

## 2️⃣ YOLO Detection

### Process Flow

```
1. Create tiles from downscaled frame (1920×804)
2. Extract tile images
3. Run YOLO on all tiles
   - Sequential: One tile at a time
   - Batched: All tiles in one forward pass
4. Convert tile-local coords to global coords
5. Store all detections with confidence scores
```

### Coordinate Transformation

```
Tile coordinates (local):
────────────────────────
Tile 2 starts at (512, 0)
Detection in tile: (100, 200, 150, 300)

Global coordinates:
────────────────────────
Add tile offset: (100+512, 200+0, 150+512, 300+0)
Result: (612, 200, 662, 300)
```

### Batch vs Sequential

**Sequential Processing:**
```python
for tile in tiles:
    result = model(tile, imgsz=640, conf=0.25)
    # Process each tile separately
```
- Simpler
- Lower memory
- Slower (multiple forward passes)

**Batch Processing:**
```python
all_results = model(tiles, imgsz=640, conf=0.25)
# Process all tiles in one forward pass
```
- Faster (1 forward pass)
- Better GPU utilization
- Requires more VRAM

---

## 3️⃣ SAHI NMS (Same as Two-Stage)

### Purpose
Merge duplicate detections from overlapping tiles

### Algorithm
```
For each pair of detections:
  1. Calculate IoU (Intersection over Union)
  2. If IoU > threshold (0.45):
     - Consider them as same object
     - Keep detection with higher confidence
     - Discard lower confidence detection
  3. Repeat until no more merges
```

### Example

```
Before SAHI NMS:
────────────────
Tile 1: gun at (100, 200, 150, 300), conf=0.85
Tile 2: gun at (612, 200, 662, 300), conf=0.82  ← Same gun!
Tile 3: knife at (1200, 400, 1280, 520), conf=0.91

After SAHI NMS:
────────────────
gun at (100, 200, 150, 300), conf=0.85
knife at (1200, 400, 1280, 520), conf=0.91

Reduction: 3 → 2 detections
```

---

## 4️⃣ ROI Refinement (Optional)

### What is ROI Refinement?

Instead of using a **separate classifier**, we **re-run YOLO** on full-resolution crops around detections.

### Why ROI Refinement?

```
Problem with Downscaled Detection:
──────────────────────────────────
Original: 3840×1608
Downscaled: 1920×804 (50% size)

Small objects become even smaller!
Details are lost in downscaling.

Solution:
──────────────────────────────────
1. Detect on downscaled (fast, finds general location)
2. Crop from FULL-RES original frame
3. Re-detect on full-res crop (accurate, sees details)
```

### ROI Refinement Process

```
Step 1: Get merged detection
────────────────────────────
Detection: gun at (100, 200, 150, 300) @ 1920×804

Step 2: Scale to full resolution
────────────────────────────
Full-res coords: (200, 400, 300, 600) @ 3840×1608

Step 3: Expand ROI for context
────────────────────────────
Original box: 100×200 pixels
Expand by 20%: +20px on each side
Expanded: (180, 380, 320, 620)

Step 4: Crop from original frame
────────────────────────────
crop = frame_orig[380:620, 180:320]
Size: 140×240 pixels (full resolution)

Step 5: Re-run YOLO on crop
────────────────────────────
result = YOLO(crop, conf=0.30)

Step 6: Verify detection
────────────────────────────
If YOLO detects object in crop:
  ✓ Keep detection (verified)
  Update coordinates and confidence
Else:
  ✗ Reject detection (false positive)
```

### Refinement Parameters

**`--roi_expand=0.2`** (20% expansion)
```
Gives context around object
Too small: Might crop object edges
Too large: Includes irrelevant background
```

**`--refine_conf=0.30`** (lower threshold)
```
Lower than initial detection (0.50)
Allows re-detection in cropped context
Higher confidence = fewer false positives
```

### Example Workflow

```
Frame arrives
     ↓
Downscale to 1920×804
     ↓
Detect: 3 detections
     ↓
SAHI NMS: 2 detections
     ↓
ROI Refinement:
  Detection 1:
    - Crop from 3840×1608
    - YOLO re-detects → ✓ Verified
    - Update coords & conf
  
  Detection 2:
    - Crop from 3840×1608
    - YOLO finds nothing → ✗ Rejected
     ↓
Final: 1 verified detection
```

### Performance Impact

```
Without ROI Refinement:
  Detection: 20-25ms
  Total: 25-30ms
  
With ROI Refinement:
  Detection: 20-25ms
  ROI crops: 2-3 detections
  Re-detection: 5-10ms (batch of crops)
  Total: 30-40ms
```

---

## 5️⃣ ByteTrack (Same as Two-Stage)

### Tracking Flow

```
Frame N:   gun at (100, 200) → Assign ID=1
Frame N+1: gun at (105, 205) → Match to ID=1 (same gun, moved)
Frame N+2: gun at (110, 210) → Match to ID=1
Frame N+3: no detection      → ID=1 still alive (persist)
Frame N+4: gun at (115, 215) → Match to ID=1 (re-identified)
```

### Key Parameters

**`--min_hits=3`**
```
Track must be detected 3 consecutive times to appear
Prevents brief false positives from showing
```

**`--track_persist=30`**
```
Keep track alive for 30 frames after last detection
Handles temporary occlusions
At 30 FPS = 1 second of persistence
```

**`--match_thresh=0.8`**
```
IoU threshold for matching detection to track
0.8 = Very strict (prevents ID switches)
```

---

## 6️⃣ Complete Pipeline Comparison

### Timeline: Detection-Only

```
T=0ms   : Frame arrives (3840×1608)
T=1ms   : Downscale to 1920×804
T=2ms   : Create ~8 tiles (640×640)
T=5ms   : YOLO detection on all tiles
T=25ms  : SAHI NMS merge
T=27ms  : Extract ROIs (if refinement enabled)
T=30ms  : YOLO re-detection on crops
T=40ms  : ByteTrack update
T=42ms  : Visualization
T=45ms  : Frame complete ✓

Target: < 33.3ms @ 30 FPS
Actual: ~45ms (MARGINAL)
```

### Timeline: Two-Stage (for comparison)

```
T=0ms   : Frame arrives (3840×1608)
T=1ms   : Downscale to 1920×804
T=2ms   : Create 8 tiles (640×640)
T=5ms   : YOLO detection (batch=8)
T=25ms  : SAHI NMS merge
T=27ms  : Extract ROIs
T=30ms  : ConvNeXT classification
T=38ms  : ByteTrack update
T=40ms  : Visualization
T=42ms  : Frame complete ✓

Target: < 33.3ms @ 30 FPS
Actual: ~35-42ms (REAL-TIME)
```

---

## 📈 Performance Comparison

### Detection-Only Pipeline

**Pros:**
```
✅ Single model (YOLO only)
✅ Simpler deployment
✅ Lower memory footprint
✅ No classifier training needed
```

**Cons:**
```
❌ Slower (45ms vs 35ms)
   - ROI refinement adds 10-15ms
   - Re-running YOLO is expensive
❌ Less accurate
   - No specialized gun/knife classifier
   - More false positives
```

### Two-Stage Pipeline

**Pros:**
```
✅ Faster (35ms vs 45ms)
   - ConvNeXT is lighter than YOLO
   - Batch classification efficient
✅ More accurate
   - Specialized classifier
   - 98%+ gun/knife accuracy
✅ Better filtering
   - Strict classification threshold
   - Reduces false positives
```

**Cons:**
```
❌ More complex
   - Two models to manage
   - Classifier training required
❌ Higher memory
   - YOLO + ConvNeXT loaded
```

---

## 🔧 Configuration

### Detection-Only Script

```bash
python tiled_tensorrt_realtime.py \
  --video video.mp4 \
  --model yolo11s_640.engine \
  --tile_size 640 \
  --overlap 128 \
  --conf 0.25 \
  --iou 0.45 \
  --camera_fps 30 \
  --downscale 0.5 \
  --batch_tiles \              # Batch process tiles
  --refine_rois \              # Enable ROI refinement
  --roi_expand 0.2 \           # 20% expansion
  --refine_conf 0.30 \         # Lower conf for refinement
  --track \                     # Enable tracking
  --min_hits 3 \               # Anti-flicker
  --track_persist 30 \         # 1 second @ 30fps
  --save_vis                   # Save video
```

### Two-Stage Script (for comparison)

```bash
python tiled_classification_realtime.py \
  --video video.mp4 \
  --detect_model yolo11s_640.engine \
  --classify_model convnext.ts \
  --tile_size 640 \
  --detect_batch 8 \
  --classify_batch 4 \
  --conf 0.50 \                # Higher YOLO threshold
  --classify_conf 0.90 \       # Strict classifier
  --iou 0.45 \
  --camera_fps 30 \
  --downscale 0.5 \
  --classify_rois \            # Use classifier instead
  --track \
  --min_hits 5 \               # Stricter anti-flicker
  --track_persist 45           # Longer persistence
```

---

## 📊 When to Use Each Approach

### Use Detection-Only When:

✅ **Simple deployment needed**
- Single model easier to manage
- Limited resources

✅ **YOLO is highly accurate**
- Well-trained on your specific classes
- Low false positive rate already

✅ **Speed not critical**
- Can tolerate 40-50ms latency
- 20-25 FPS acceptable

✅ **Memory constrained**
- Can't load two models
- Embedded devices

### Use Two-Stage When:

✅ **Maximum accuracy required**
- Gun vs knife distinction critical
- False positives unacceptable

✅ **Real-time required**
- Need 30+ FPS consistently
- < 33ms latency target

✅ **Classification expertise**
- Can train specialized classifier
- Have labeled crop dataset

✅ **Resources available**
- GPU memory for two models
- Can deploy complex pipeline

---

## 🎬 Visual Comparison

### Detection-Only Flow
```
VIDEO FRAME
     ↓
  Downscale
     ↓
  Tile (8x)
     ↓
┌──────────┐
│   YOLO   │ ← Single model
└──────────┘
     ↓
  SAHI NMS
     ↓
┌──────────┐
│   YOLO   │ ← Same model again (refinement)
└──────────┘
     ↓
  ByteTrack
     ↓
  OUTPUT
```

### Two-Stage Flow
```
VIDEO FRAME
     ↓
  Downscale
     ↓
  Tile (8x)
     ↓
┌──────────┐
│   YOLO   │ ← Detection model
└──────────┘
     ↓
  SAHI NMS
     ↓
┌──────────┐
│ ConvNeXT │ ← Different classification model
└──────────┘
     ↓
  ByteTrack
     ↓
  OUTPUT
```

---

## 💡 Key Takeaways

### Detection-Only Pipeline

**Best for:**
- Prototyping and testing
- Simple deployments
- When YOLO accuracy is sufficient
- Resource-constrained environments

**Characteristics:**
- **Simpler**: One model to manage
- **Slower**: ROI refinement is expensive
- **Less accurate**: No specialized classification
- **Lower memory**: Single model loaded

### Recommendation

**Start with Detection-Only** to:
- Validate the tiling approach
- Understand performance characteristics
- Identify false positive patterns

**Upgrade to Two-Stage** when:
- Detection-only has too many FPs
- Need gun/knife distinction accuracy
- Speed becomes critical
- Resources are available

---

## 🔄 Migration Path

### From Detection-Only to Two-Stage

1. **Collect crop dataset**
   - Save ROI crops from detection-only pipeline
   - Label as gun/knife
   - 2000+ samples recommended

2. **Train ConvNeXT classifier**
   ```bash
   python train_convnext.py \
     --data crops/ \
     --epochs 20
   ```

3. **Export classifier**
   ```bash
   python export_torch_compile.py \
     --model_path checkpoint/ \
     --batch_size 4
   ```

4. **Switch scripts**
   - Replace `tiled_tensorrt_realtime.py`
   - With `tiled_classification_realtime.py`
   - Update config parameters

5. **Tune thresholds**
   - Start with `--conf 0.50`
   - Set `--classify_conf 0.90`
   - Adjust based on results

---

**Documentation for legacy detection-only approach using single YOLO model with optional ROI refinement**

