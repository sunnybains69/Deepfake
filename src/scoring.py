"""
Step 4: turn per-frame fake probabilities into one video decision.

A plain mean dilutes a few strongly manipulated frames among many
ordinary-looking ones. "topk" instead averages only the most
suspicious fraction of frames. Because topk raises the score of REAL
videos too, the decision threshold must be chosen on validation
videos (see calibrate_threshold.py), not fixed at 0.5.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

AGGREGATION_METHODS = ("topk", "mean", "median", "max")
DEFAULT_THRESHOLD_PATH = "models/threshold.json"


def aggregate_fake_scores(
    probabilities: list[float],
    method: str = "topk",
    top_fraction: float = 0.25,
) -> float:
    values = np.asarray(probabilities, dtype=float)

    if values.size == 0:
        raise ValueError("No frame probabilities to aggregate.")

    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    if method == "max":
        return float(values.max())
    if method == "topk":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")
        k = max(1, math.ceil(values.size * top_fraction))
        return float(np.sort(values)[-k:].mean())

    raise ValueError(
        f"Unknown aggregation '{method}'. Use one of {AGGREGATION_METHODS}."
    )


def choose_threshold(scores: list[float], is_fake: list[int]) -> dict:
    """
    Pick the threshold that maximises balanced accuracy
    ((fake recall + real recall) / 2) on validation videos.
    A video is called fake when score >= threshold.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels = np.asarray(is_fake, dtype=int)

    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("Need both real and fake videos to calibrate.")

    unique = np.unique(scores_arr)
    # Candidates: midpoints between neighbouring scores, plus ends.
    candidates = np.concatenate(
        [[0.0], (unique[:-1] + unique[1:]) / 2, [unique[-1] + 1e-9]]
    )

    best = None
    for threshold in candidates:
        metrics = video_metrics(scores_arr, labels, threshold)
        key = (metrics["balanced_accuracy"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, metrics)

    return best[1]


def video_metrics(scores, is_fake, threshold: float) -> dict:
    scores_arr = np.asarray(scores, dtype=float)
    labels = np.asarray(is_fake, dtype=int)
    predicted_fake = scores_arr >= threshold

    tp = int(np.sum(predicted_fake & (labels == 1)))
    fn = int(np.sum(~predicted_fake & (labels == 1)))
    tn = int(np.sum(~predicted_fake & (labels == 0)))
    fp = int(np.sum(predicted_fake & (labels == 0)))

    fake_recall = tp / (tp + fn) if tp + fn else 0.0
    real_recall = tn / (tn + fp) if tn + fp else 0.0

    return {
        "threshold": float(threshold),
        "balanced_accuracy": (fake_recall + real_recall) / 2,
        "accuracy": (tp + tn) / labels.size if labels.size else 0.0,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "true_fake": tp,
        "missed_fake": fn,
        "true_real": tn,
        "false_alarm": fp,
    }


def model_fingerprint(model_path: str) -> str:
    """SHA-256 of the model file, so a retrained model is detected."""
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_threshold(
    model_path: str,
    method: str,
    top_fraction: float,
    threshold_path: str = DEFAULT_THRESHOLD_PATH,
) -> tuple[float, bool]:
    """
    Return (threshold, calibrated?). Falls back to 0.5 if no
    calibration file exists or it was made for a different model
    file (e.g. after retraining) or aggregation setting.
    """
    path = Path(threshold_path)

    if not path.exists():
        return 0.5, False

    data = json.loads(path.read_text())

    matches = (
        data.get("model_sha256") == model_fingerprint(model_path)
        and data.get("aggregation") == method
        and (method != "topk"
             or math.isclose(data.get("top_fraction", -1), top_fraction))
    )

    if not matches:
        print(
            f"Warning: {path} was calibrated for a different model "
            "file or aggregation setting; using threshold 0.5. "
            "Re-run calibrate_threshold.py after retraining."
        )
        return 0.5, False

    return float(data["threshold"]), True
