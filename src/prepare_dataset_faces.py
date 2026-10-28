
from pathlib import Path
from collections import defaultdict
import random
import cv2

# Project directories
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "faceforensics"

# New output directory - preserves existing dataset
OUTPUT = ROOT / "data" / "faceforensics_faces"

REAL_DIR = SOURCE / "original_sequences" / "youtube" / "c23" / "videos"
FAKE_DIR = SOURCE / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"

FRAMES_PER_VIDEO = 10
SEED = 42
IMAGE_SIZE = 224
FACE_MARGIN = 0.25

# OpenCV face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detector.")

# Find videos
real_videos = {
    p.stem: p for p in REAL_DIR.glob("*.mp4")
}
fake_videos = sorted(FAKE_DIR.glob("*.mp4"))

if not real_videos or not fake_videos:
    raise RuntimeError("Could not find the source videos.")

# Group related original and manipulated videos
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
        raise ValueError(
            f"Missing original for {video.name}"
        )

    union(a, b)

groups = defaultdict(
    lambda: {"real": [], "fake": []}
)

for name, path in real_videos.items():
    groups[find(name)]["real"].append(path)

for path in fake_videos:
    a, _ = path.stem.split("_")
    groups[find(a)]["fake"].append(path)

# Reproduce the original train/validation/test split
group_list = list(groups.values())
random.Random(SEED).shuffle(group_list)

n = len(group_list)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "train": group_list[:train_end],
    "val": group_list[train_end:val_end],
    "test": group_list[val_end:]
}

def crop_face(frame):
    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Add margin around the face
    margin = int(max(w, h) * FACE_MARGIN)

    height, width = frame.shape[:2]

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return face

def extract_faces(video_path, output_dir):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Cannot open: {video_path}")
        return 0

    total = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        capture.release()
        return 0

    saved = 0

    for i in range(FRAMES_PER_VIDEO):
        frame_index = int(
            (i + 0.5) * total / FRAMES_PER_VIDEO
        )
        frame_index = min(
            frame_index, total - 1
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        if not success:
            continue

        face = crop_face(frame)

        if face is None:
            continue

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            output_dir
            / f"{video_path.stem}_frame{i:02d}.jpg"
        )

        if cv2.imwrite(str(filename), face):
            saved += 1

    capture.release()
    return saved

# Protect against accidental overwriting
if OUTPUT.exists() and any(OUTPUT.rglob("*.jpg")):
    raise RuntimeError(
        f"{OUTPUT} already contains images."
    )

for split_name, group_items in splits.items():
    counts = {"real": 0, "fake": 0}

    for group in group_items:
        for label in ("real", "fake"):
            for video_path in group[label]:
                destination = (
                    OUTPUT / split_name / label
                )

                counts[label] += extract_faces(
                    video_path,
                    destination
                )

    print(
        f"{split_name}: "
        f"real={counts['real']} images, "
        f"fake={counts['fake']} images"
    )

print(
    f"\nFace dataset saved to: {OUTPUT}"
)
