import subprocess
import os
import sys
import argparse
from pathlib import Path

def check_ffmpeg():
    """Checks if ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def get_output_filename(input_path, start, end):
    """Generates an output filename if one isn't provided."""
    path = Path(input_path)
    # create a name like: original_name_00-01-30_to_00-02-00.mp4
    safe_start = start.replace(":", "-")
    safe_end = end.replace(":", "-")
    new_name = f"{path.stem}_{safe_start}_to_{safe_end}{path.suffix}"
    return path.parent / new_name

def slice_video(input_file, start_time, end_time, output_file=None, precise=False):
    """
    Slices video using ffmpeg.
    
    Args:
        precise (bool): If True, re-encodes for frame-perfect cut. 
                        If False, uses stream copy (fast but snaps to keyframes).
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Determine output filename automatically if not given
    if not output_file:
        output_file = get_output_filename(input_file, start_time, end_time)

    # Base command
    # -ss before -i is faster (input seeking)
    cmd = ['ffmpeg', '-y', '-ss', start_time, '-i', input_file, '-to', end_time]

    if precise:
        # Re-encode for accuracy (slower)
        print("Mode: Precise (Re-encoding)... this may take a moment.")
        # libx264 for video, aac for audio are standard safe choices
        cmd.extend(['-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental'])
    else:
        # Stream copy for speed
        print("Mode: Fast (Stream Copy)...")
        cmd.extend(['-c', 'copy'])

    cmd.append(str(output_file))

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success! Slice saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ffmpeg execution: {e}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Slice a video using ffmpeg wrapper.")
    
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("start", help="Start time (e.g., 00:01:30 or 90)")
    parser.add_argument("end", help="End time (e.g., 00:01:45 or 105)")
    parser.add_argument("--output", "-o", help="Path to output file (optional)", default=None)
    parser.add_argument("--precise", "-p", action="store_true", help="Use re-encoding for frame-perfect accuracy (slower)")

    args = parser.parse_args()

    if not check_ffmpeg():
        print("❌ Error: ffmpeg is not installed. Run 'sudo apt install ffmpeg' first.")
        sys.exit(1)

    slice_video(args.input, args.start, args.end, args.output, args.precise)

"""
Remove multiple time intervals from a video and keep the rest intact.

- Copies video stream (no quality loss)
- Re-encodes audio to AAC (required for FLAC-in-MP4)
- Supports multiple removal intervals


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
    """'''HH:MM:SS(.ms) -> seconds'''"""
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
"""
