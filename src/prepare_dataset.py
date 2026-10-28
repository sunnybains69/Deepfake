
from pathlib import Path
from collections import defaultdict
import random
import cv2

# Project directories
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "faceforensics"
OUTPUT = ROOT / "data" / "faceforensics_prepared"

REAL_DIR = SOURCE / "original_sequences" / "youtube" / "c23" / "videos"
FAKE_DIR = SOURCE / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"

FRAMES_PER_VIDEO = 10
SEED = 42

# Identify related original videos using connected components
real_videos = {p.stem: p for p in REAL_DIR.glob("*.mp4")}
fake_videos = sorted(FAKE_DIR.glob("*.mp4"))

parent = {name: name for name in real_videos}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for video in fake_videos:
    a, b = video.stem.split("_")
    if a not in parent or b not in parent:
        raise ValueError(f"Missing original for {video.name}")
    union(a, b)

groups = defaultdict(lambda: {"real": [], "fake": []})

for name, path in real_videos.items():
    groups[find(name)]["real"].append(path)

for path in fake_videos:
    a, _ = path.stem.split("_")
    groups[find(a)]["fake"].append(path)

# Randomise related groups, not individual frames
group_list = list(groups.values())
random.Random(SEED).shuffle(group_list)

# Approximately 70/15/15 group split
n = len(group_list)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "train": group_list[:train_end],
    "val": group_list[train_end:val_end],
    "test": group_list[val_end:]
}

def extract_frames(video_path, output_dir):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Cannot open: {video_path}")
        return 0

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        print(f"No frames: {video_path}")
        return 0

    saved = 0

    for i in range(FRAMES_PER_VIDEO):
        frame_index = int(
            (i + 0.5) * total / FRAMES_PER_VIDEO
        )
        frame_index = min(frame_index, total - 1)

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        if not success:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{video_path.stem}_frame{i:02d}.jpg"

        if cv2.imwrite(str(filename), frame):
            saved += 1

    capture.release()
    return saved

# Prevent mixing with an earlier prepared dataset
if OUTPUT.exists() and any(OUTPUT.rglob("*.jpg")):
    raise RuntimeError(
        f"{OUTPUT} already contains images. "
        "Move or rename it before running again."
    )

for split_name, group_items in splits.items():
    counts = {"real": 0, "fake": 0}
    videos = {"real": 0, "fake": 0}

    for group in group_items:
        for label in ("real", "fake"):
            for video_path in group[label]:
                destination = OUTPUT / split_name / label
                counts[label] += extract_frames(
                    video_path, destination
                )
                videos[label] += 1

    print(
        f"{split_name}: "
        f"real={counts['real']} images "
        f"({videos['real']} videos), "
        f"fake={counts['fake']} images "
        f"({videos['fake']} videos)"
    )

print(f"\nPrepared dataset saved to: {OUTPUT}")
