import subprocess
from pathlib import Path

VIDEO = Path("/workspace/input_videos/25_december_videos/parking_lot_front.mp4")
TIMESTAMPS = Path("timestamps.txt")
OUT = Path("parking_lot_front_no_weapons.mp4")
TMP = Path("tmp_segments")

TMP.mkdir(exist_ok=True)

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# 1. Read weapon intervals
intervals = []
with open(TIMESTAMPS) as f:
    for line in f:
        if line.strip():
            s, e = line.split()
            intervals.append((s, e))

# 2. Get video duration
result = subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries",
     "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
     str(VIDEO)]
)
duration = float(result.strip())

def to_sec(t):
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

weapon = [(to_sec(s), to_sec(e)) for s, e in intervals]
weapon.sort()

# 3. Build keep intervals
keep = []
prev = 0.0
for s, e in weapon:
    if s > prev:
        keep.append((prev, s))
    prev = e
if prev < duration:
    keep.append((prev, duration))

# 4. Slice kept segments
segment_files = []
for i, (s, e) in enumerate(keep):
    out = TMP / f"keep_{i:02d}.mp4"
    segment_files.append(out)

    run([
        "ffmpeg", "-y",
        "-ss", str(s), "-to", str(e),
        "-i", str(VIDEO),
        "-c", "copy",
        str(out)
    ])

# 5. Concatenate
list_file = TMP / "concat.txt"
with open(list_file, "w") as f:
    for seg in segment_files:
        f.write(f"file '{seg.absolute()}'\n")

run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(list_file),
    "-c", "copy",
    str(OUT)
])

print("\n✅ Weapon segments removed.")
print("Output:", OUT)
