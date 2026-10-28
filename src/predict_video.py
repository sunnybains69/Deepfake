import argparse
from pathlib import Path

import cv2
from PIL import Image

from .model import load_model
from .predict import predict_pil_image
from .scoring import (
    AGGREGATION_METHODS,
    DEFAULT_THRESHOLD_PATH,
    aggregate_fake_scores,
    load_threshold,
)
from .utils import get_device

# Scores this close to the threshold are reported as "uncertain".
UNCERTAIN_MARGIN = 0.05


def score_video_frames(
    video_path: str,
    model,
    class_to_idx: dict,
    device,
    sample_every: int = 15,
    max_frames: int = 60,
    crop_faces: bool = True,
    verbose: bool = True,
) -> tuple[list[float], int]:
    """
    Return (per-frame fake probabilities, frames skipped because
    no face was found). Shared by predict_video and
    calibrate_threshold so both score videos identically.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    if sample_every < 1 or max_frames < 1:
        raise ValueError(
            "sample_every and max_frames must be at least 1."
        )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fake_probabilities = []
    frame_number = 0
    skipped_faces = 0

    try:
        while len(fake_probabilities) < max_frames:
            success, frame = capture.read()

            if not success:
                break

            if frame_number % sample_every == 0:
                # Convert OpenCV BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)

                try:
                    result = predict_pil_image(
                        image,
                        model,
                        class_to_idx,
                        device,
                        crop_faces=crop_faces,
                    )
                except ValueError as error:
                    if (
                        crop_faces
                        and str(error) == "No face detected in the image."
                    ):
                        skipped_faces += 1
                        frame_number += 1
                        continue
                    raise

                fake_probabilities.append(result["fake"])

                if verbose:
                    print(
                        f"Frame {frame_number}: "
                        f"FAKE={result['fake']:.2%}, "
                        f"REAL={result['real']:.2%}"
                    )

            frame_number += 1

    finally:
        capture.release()

    return fake_probabilities, skipped_faces


def predict_video(
    video_path: str,
    model_path: str = "models/best_model_faces.pth",
    sample_every: int = 15,
    max_frames: int = 60,
    crop_faces: bool = True,
    aggregation: str = "topk",
    top_fraction: float = 0.25,
    threshold_path: str = DEFAULT_THRESHOLD_PATH,
    model=None,
    checkpoint=None,
    device=None,
):
    # Load the model unless the caller (e.g. the app) passes one in
    if model is None or checkpoint is None:
        device = get_device()
        model, checkpoint = load_model(model_path, device)
    elif device is None:
        device = next(model.parameters()).device

    model.eval()
    class_to_idx = checkpoint["class_to_idx"]

    fake_probabilities, skipped_faces = score_video_frames(
        video_path,
        model,
        class_to_idx,
        device,
        sample_every=sample_every,
        max_frames=max_frames,
        crop_faces=crop_faces,
    )

    if not fake_probabilities:
        raise RuntimeError("No usable frames were found in the video.")

    # Combine frames (step 4) and compare with calibrated threshold
    video_score = aggregate_fake_scores(
        fake_probabilities, method=aggregation, top_fraction=top_fraction,
    )
    threshold, calibrated = load_threshold(
        model_path, aggregation, top_fraction, threshold_path,
    )

    if abs(video_score - threshold) <= UNCERTAIN_MARGIN:
        prediction = "uncertain"
    elif video_score > threshold:
        prediction = "fake"
    else:
        prediction = "real"

    print("\nVideo analysis completed")
    print(f"Prediction: {prediction.upper()}")
    print(
        f"Video fake score ({aggregation}): {video_score:.2%} "
        f"| threshold: {threshold:.2%} "
        f"({'calibrated' if calibrated else 'NOT calibrated, default'})"
    )
    print(f"Frames analysed: {len(fake_probabilities)}")
    print(f"Frames skipped (no face detected): {skipped_faces}")

    return {
        "prediction": prediction,
        # Kept for the app: the video-level fake score and its
        # complement. These are scores, not calibrated probabilities.
        "fake_probability": video_score,
        "real_probability": 1.0 - video_score,
        "threshold": threshold,
        "threshold_calibrated": calibrated,
        "aggregation": aggregation,
        "frame_fake_probabilities": fake_probabilities,
        "sampled_frames": len(fake_probabilities),
        "skipped_faces": skipped_faces,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a video is real, fake, or uncertain."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--model-path", default="models/best_model_faces.pth")
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument(
        "--no-crop-faces",
        action="store_true",
        help="Disable face detection and cropping.",
    )
    parser.add_argument(
        "--aggregation", choices=AGGREGATION_METHODS, default="topk",
    )
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--threshold-path", default=DEFAULT_THRESHOLD_PATH)
    args = parser.parse_args()

    predict_video(
        video_path=args.video,
        model_path=args.model_path,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        crop_faces=not args.no_crop_faces,
        aggregation=args.aggregation,
        top_fraction=args.top_fraction,
        threshold_path=args.threshold_path,
    )


if __name__ == "__main__":
    main()
