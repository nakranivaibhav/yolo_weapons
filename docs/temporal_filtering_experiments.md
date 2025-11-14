## How Temporal Filtering Works right now

We use **ByteTrack** to solve: when weapons move fast, they get blurry and the detector loses them.

### The Problem
When someone moves quickly, motion blur happens. Without tracking:
- The system loses track of the weapon when it gets blurry
- Creates duplicate alerts when it shows up again
- Flickers

### How We Fixed It: Two Layers

#### Layer 1: Memory Buffer
**What it does**: Remembers objects even when they're temporarily hidden

**Simple example**: 
- Object is occluded.
- With 1-second memory → Model gave us re detections.
- 4-second memory was more reliable.

#### Layer 2: Minimum hits for an object above a threshold.
- Does not trigger an alert the weapon 5 frames in a row

- Sometimes a shadow or a phone case looks like a gun for 1 frame. The 5-frame rule filters out this random noise.

---

## Test Results: Different Memory Lengths

We tested 1-4 second memory buffers to see what works best.

### Setup
- 4K video at 30 FPS
- YOLO11 detecting weapons
- ByteTrack with different buffer sizes
- 5-frame rule active
- Need to stay under 33.3ms per frame to keep 30 FPS

### Results

| Memory Duration | Inference Time | How Much Speed We're Using | Dropped Detections |
|-----------------|----------------|---------------------------|-------------------|
| 1 second        | 18.1ms        | 54%                       | 0.3%              |
| 2 seconds       | 17.5ms        | 53%                       | 0.3%              |
| 3 seconds       | 17.3ms        | 52%                       | 0.3%              |
| 4 seconds       | 17.8ms        | 53%                       | 0.2%              |

## What We Recommend

After all the testing, here's what works best:

```bash
--track                  # Turn on ByteTrack
--track_persist 120      # 4-second memory
--min_hits 5-7            # Need 5-7 frames in a row
```

**Why these settings:**
- 4-second memory: Best stability.
- 5/7-frame rule: Filters out flicker and noise
- Together: Stable detections at ~18ms per frame (plenty fast for 30 FPS)

---