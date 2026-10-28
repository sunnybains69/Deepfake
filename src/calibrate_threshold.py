"""
Step 4: choose the video-level decision threshold on VALIDATION
videos, then report honest results on the held-out TEST videos.

Uses split_manifest.json written by prepare_split.py, so the videos
scored here are never ones the model was trained on.

    python -m src.calibrate_threshold \
        --manifest data_by_video/split_manifest.json \
        --model-path models/best_model_faces.pth

Writes models/threshold.json, which predict_video.py and the app
pick up automatically.
"""

import argparse
import json
from pathlib import Path

from .model import load_model
from .predict_video import score_video_frames
from .scoring import (
    AGGREGATION_METHODS,
    DEFAULT_THRESHOLD_PATH,
    aggregate_fake_scores,
    choose_threshold,
    model_fingerprint,
    video_metrics,
)
from .utils import get_device


def score_split(videos, model, class_to_idx, device, args):
    scores, labels, rows = [], [], []

    for index, video in enumerate(videos, start=1):
        frame_probs, skipped = score_video_frames(
            video["path"],
            model,
            class_to_idx,
            device,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            crop_faces=True,
            verbose=False,
        )

        if not frame_probs:
            print(f"  [{index}/{len(videos)}] no faces, skipped: {video['path']}")
            continue

        score = aggregate_fake_scores(
            frame_probs, method=args.aggregation, top_fraction=args.top_fraction,
        )
        is_fake = int(video["label"] == "fake")
        scores.append(score)
        labels.append(is_fake)
        rows.append({"path": video["path"], "label": video["label"], "score": score})
        print(
            f"  [{index}/{len(videos)}] {video['label']:4s} "
            f"score={score:.3f}  {Path(video['path']).name}"
        )

    return scores, labels, rows


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"  Threshold:         {metrics['threshold']:.3f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Fake recall:       {metrics['fake_recall']:.3f}  "
          f"(missed fakes = {metrics['missed_fake']})")
    print(f"  Real recall:       {metrics['real_recall']:.3f}  "
          f"(false alarms = {metrics['false_alarm']})")
    print("  Confusion matrix (rows = actual, cols = predicted):")
    print("                pred real  pred fake")
    print(f"    real        {metrics['true_real']:9d}  {metrics['false_alarm']:9d}")
    print(f"    fake        {metrics['missed_fake']:9d}  {metrics['true_fake']:9d}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the video-level fake threshold."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", default="models/best_model_faces.pth")
    parser.add_argument(
        "--aggregation", choices=AGGREGATION_METHODS, default="topk",
    )
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--output", default=DEFAULT_THRESHOLD_PATH)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    by_split = {"val": [], "test": []}
    for video in manifest["videos"]:
        if video["split"] in by_split:
            by_split[video["split"]].append(video)

    device = get_device()
    model, checkpoint = load_model(args.model_path, device)
    class_to_idx = checkpoint["class_to_idx"]

    print(f"Scoring {len(by_split['val'])} validation videos...")
    val_scores, val_labels, val_rows = score_split(
        by_split["val"], model, class_to_idx, device, args,
    )

    best = choose_threshold(val_scores, val_labels)
    print_metrics("Validation (threshold chosen here):", best)

    result = {
        "model_path": args.model_path,
        "model_sha256": model_fingerprint(args.model_path),
        "aggregation": args.aggregation,
        "top_fraction": args.top_fraction,
        "sample_every": args.sample_every,
        "max_frames": args.max_frames,
        "threshold": best["threshold"],
        "validation": best,
        "validation_videos": val_rows,
    }

    if by_split["test"]:
        print(f"\nScoring {len(by_split['test'])} test videos...")
        test_scores, test_labels, test_rows = score_split(
            by_split["test"], model, class_to_idx, device, args,
        )
        test = video_metrics(test_scores, test_labels, best["threshold"])
        print_metrics("TEST (held out - report these numbers):", test)
        result["test"] = test
        result["test_videos"] = test_rows

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"\nSaved threshold to {output}")


if __name__ == "__main__":
    main()
