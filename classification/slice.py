"""
Remove multiple time intervals from a video and keep the rest intact.

- Copies video stream (no quality loss)
- Re-encodes audio to AAC (required for FLAC-in-MP4)
- Supports multiple removal intervals
"""

import subprocess
from pathlib import Path

# ---------------- USER CONFIG ----------------

VIDEO = Path("/workspace/input_videos/25_december_videos/outside_left.mp4")          # input video
TIMESTAMPS = Path("timestamps.txt")             # intervals to REMOVE
OUTPUT = Path("outside_left_no_weapons.mp4")

TMP_DIR = Path("tmp_segments")

# ---------------------------------------------

TMP_DIR.mkdir(exist_ok=True)

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def to_seconds(t):
    """HH:MM:SS(.ms) -> seconds"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# ---------------- Read weapon intervals ----------------

weapon_intervals = []
with open(TIMESTAMPS) as f:
    for line in f:
        if line.strip():
            start, end = line.split()
            weapon_intervals.append((to_seconds(start), to_seconds(end)))

weapon_intervals.sort()

# ---------------- Get video duration (robust) ----------------

def get_duration(video):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video)
        ])
        return float(out.strip())
    except Exception:
        print("⚠️ ffprobe failed, using large fallback duration")
        return 10_000.0  # safe fallback

duration = get_duration(VIDEO)

# ---------------- Build KEEP intervals ----------------

keep_intervals = []
prev_end = 0.0

for start, end in weapon_intervals:
    if start > prev_end:
        keep_intervals.append((prev_end, start))
    prev_end = end

if prev_end < duration:
    keep_intervals.append((prev_end, duration))

print("\nKEEP intervals:")
for s, e in keep_intervals:
    print(f"  {s:.2f} → {e:.2f}")

# ---------------- Slice KEEP segments ----------------

segment_files = []

for idx, (start, end) in enumerate(keep_intervals):
    out_seg = TMP_DIR / f"keep_{idx:02d}.mp4"
    segment_files.append(out_seg)

    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(VIDEO),
        "-c:v", "copy",      # keep video quality
        "-c:a", "aac",       # FIX for FLAC audio
        str(out_seg)
    ])

# ---------------- Create concat list ----------------

concat_file = TMP_DIR / "concat.txt"
with open(concat_file, "w") as f:
    for seg in segment_files:
        f.write(f"file '{seg.resolve()}'\n")

# ---------------- Concatenate ----------------

run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(concat_file),
    "-c:v", "copy",
    "-c:a", "aac",
    str(OUTPUT)
])

print("\n✅ Weapon segments removed successfully")
print("Output video:", OUTPUT)
