"""
Build a face-crop dataset that is split BY VIDEO (step 1).

Why: if frames from the same video appear in both train/ and val/,
the model can memorise the person and background instead of learning
manipulation artefacts. Validation scores then look perfect
(best_model.pth reached F1 = 1.00) but new fake videos are missed.

This script:
  * takes folders of REAL and FAKE videos,
  * assigns every video (and, by default, every person/identity)
    to exactly ONE of train / val / test,
  * samples frames from each video and crops the face using the
    same crop_face() function that the app uses at inference,
  * writes data/<split>/<fake|real>/*.png for train.py,
  * writes split_manifest.json listing which videos went where
    (used by calibrate_threshold.py and for your dissertation).

Example (FaceForensics++ layout):

    python -m src.prepare_split \
        --real-dir FF++/original_sequences/youtube/c23/videos \
        --fake-dir FF++/manipulated_sequences/Deepfakes/c23/videos \
        --fake-dir FF++/manipulated_sequences/FaceSwap/c23/videos \
        --output-dir data_by_video
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SPLITS = ("train", "val", "test")


# --------------------------------------------------
# Grouping and splitting (pure Python, no OpenCV)
# --------------------------------------------------

def source_tag(folder: Path, index: int, label: str) -> str:
    """
    Short name for an input folder, used in image file names.
    FF++ .../Deepfakes/c23/videos -> "fake0-Deepfakes".
    """
    generic = {"videos", "c23", "c40", "raw"}
    name = next(
        (part for part in reversed(folder.resolve().parts)
         if part.lower() not in generic),
        "videos",
    )
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{label}{index}-{name}")


def find_videos(folders: list[str], label: str) -> list[dict]:
    videos = []

    for index, folder in enumerate(folders):
        root = Path(folder)

        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {root}")

        tag = source_tag(root, index, label)

        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(
                    {"path": str(path), "label": label, "source": tag}
                )

    return videos


def identity_ids(video: dict) -> list[str]:
    """
    Return the identity IDs that appear in a video.

    FaceForensics++ naming:
      real  000.mp4      -> person 000
      fake  000_003.mp4  -> video of person 000 with face of 003
    A fake therefore involves TWO identities; both are returned so
    that neither person can appear in another split.

    For other datasets, any file name without the NNN or NNN_MMM
    pattern is treated as its own identity.
    """
    stem = Path(video["path"]).stem
    match = re.fullmatch(r"(\d+)(?:_(\d+))?", stem)

    if match:
        return [part for part in match.groups() if part is not None]

    return [f"{video['label']}:{stem}"]


def group_videos(videos: list[dict], by_identity: bool) -> list[list[dict]]:
    """
    Group videos that must stay in the same split.

    by_identity=False: each video is its own group.
    by_identity=True:  videos sharing any identity are merged
                       (union-find), so a person never appears in
                       more than one split.
    """
    if not by_identity:
        return [[video] for video in videos]

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for video in videos:
        ids = identity_ids(video)
        for other in ids[1:]:
            union(ids[0], other)
        find(ids[0])

    groups: dict[str, list[dict]] = defaultdict(list)

    for video in videos:
        groups[find(identity_ids(video)[0])].append(video)

    return [groups[key] for key in sorted(groups)]


def split_groups(
    groups: list[list[dict]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Randomly assign whole groups to train / val / test."""
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Fractions must be between 0 and 1.")

    if val_fraction + test_fraction >= 1:
        raise ValueError("val + test fractions must be below 1.")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)

    total_videos = sum(len(group) for group in shuffled)
    targets = {
        "val": val_fraction * total_videos,
        "test": test_fraction * total_videos,
    }

    result: dict[str, list[dict]] = {split: [] for split in SPLITS}

    for group in shuffled:
        if len(result["val"]) < targets["val"]:
            split = "val"
        elif len(result["test"]) < targets["test"]:
            split = "test"
        else:
            split = "train"

        for video in group:
            result[split].append({**video, "split": split})

    return result


def check_no_leakage(split_videos: dict[str, list[dict]], by_identity: bool) -> None:
    """Fail loudly if any video or identity appears in two splits."""
    seen: dict[str, str] = {}

    for split, videos in split_videos.items():
        for video in videos:
            keys = [video["path"]]
            if by_identity:
                keys += [f"id:{i}" for i in identity_ids(video)]

            for key in keys:
                if key in seen and seen[key] != split:
                    raise RuntimeError(
                        f"Leakage: {key} is in both "
                        f"{seen[key]} and {split}."
                    )
                seen[key] = split


# --------------------------------------------------
# Frame extraction (uses OpenCV + the inference crop)
# --------------------------------------------------

def frame_indices(frame_count: int, frames_per_video: int) -> list[int]:
    """Evenly spaced frame numbers across the whole video."""
    if frame_count <= 0:
        return []

    if frame_count <= frames_per_video:
        return list(range(frame_count))

    step = frame_count / frames_per_video
    return [int(step * i + step / 2) for i in range(frames_per_video)]


def extract_faces(
    video: dict,
    output_dir: Path,
    frames_per_video: int,
) -> tuple[int, int]:
    """Save face crops for one video. Returns (saved, skipped)."""
    import cv2
    from PIL import Image

    from .predict import crop_face

    capture = cv2.VideoCapture(video["path"])

    if not capture.isOpened():
        print(f"  ! Could not open {video['path']}")
        return 0, 0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(frame_indices(frame_count, frames_per_video))

    destination = output_dir / video["split"] / video["label"]
    destination.mkdir(parents=True, exist_ok=True)

    # Unique, readable prefix, e.g. "fake0-Deepfakes__000_003"
    path = Path(video["path"])
    prefix = re.sub(
        r"[^A-Za-z0-9_.-]", "_", f"{video['source']}__{path.stem}"
    )

    saved = skipped = 0
    frame_number = 0

    try:
        while wanted and frame_number <= max(wanted):
            success, frame = capture.read()
            if not success:
                break

            if frame_number in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = crop_face(Image.fromarray(rgb))

                if face is None:
                    skipped += 1
                else:
                    # PNG avoids adding extra JPEG compression that
                    # the model would not see at inference time.
                    face.save(destination / f"{prefix}_f{frame_number:05d}.png")
                    saved += 1

            frame_number += 1
    finally:
        capture.release()

    return saved, skipped


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create a leakage-free, video-level train/val/test split."
    )
    parser.add_argument(
        "--real-dir", action="append", required=True,
        help="Folder of real videos (repeat for several folders).",
    )
    parser.add_argument(
        "--fake-dir", action="append", required=True,
        help="Folder of fake videos (repeat for each manipulation method).",
    )
    parser.add_argument("--output-dir", default="data_by_video")
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-by", choices=["identity", "video"], default="identity",
        help="identity (recommended): a person never appears in two "
             "splits. video: only whole videos are kept together.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} is not empty. Use a new folder so old, "
            "leaked frames cannot mix with the new split."
        )

    by_identity = args.split_by == "identity"

    videos = (
        find_videos(args.real_dir, "real")
        + find_videos(args.fake_dir, "fake")
    )

    if not videos:
        raise SystemExit("No videos found.")

    groups = group_videos(videos, by_identity=by_identity)
    split_videos = split_groups(
        groups, args.val_fraction, args.test_fraction, args.seed,
    )
    check_no_leakage(split_videos, by_identity=by_identity)

    print(f"Found {len(videos)} videos in {len(groups)} groups.")

    largest = max(len(group) for group in groups)
    if by_identity and largest > 0.2 * len(videos):
        print(
            f"  ! Warning: one identity group holds {largest} of "
            f"{len(videos)} videos, so the split may be lopsided. "
            "If your file names are not FaceForensics++ style, "
            "try --split-by video."
        )

    for split in SPLITS:
        labels = [video["label"] for video in split_videos[split]]
        print(
            f"  {split:5s}: {len(labels):5d} videos "
            f"(real={labels.count('real')}, fake={labels.count('fake')})"
        )
        if split_videos[split] and len(set(labels)) < 2:
            print(f"  ! Warning: {split} contains only one class.")

    totals = defaultdict(lambda: [0, 0])

    for split in SPLITS:
        for index, video in enumerate(split_videos[split], start=1):
            print(f"[{split} {index}/{len(split_videos[split])}] {video['path']}")
            saved, skipped = extract_faces(
                video, output_dir, args.frames_per_video,
            )
            video["frames_saved"] = saved
            video["frames_skipped"] = skipped
            totals[(split, video["label"])][0] += saved
            totals[(split, video["label"])][1] += skipped

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split_by": args.split_by,
        "frames_per_video": args.frames_per_video,
        "seed": args.seed,
        "videos": [v for split in SPLITS for v in split_videos[split]],
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nFace images saved (skipped = no face found):")
    for (split, label), (saved, skipped) in sorted(totals.items()):
        print(f"  {split:5s} {label:4s}: {saved:6d} saved, {skipped:5d} skipped")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
