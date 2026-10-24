import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .model import load_model
from .predict import predict_pil_image
from .utils import get_device


def predict_video(
    video_path: str,
    model_path: str,
    sample_every: int = 15,
    max_frames: int = 60,
):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    device = get_device()
    model, checkpoint = load_model(model_path, device)
    class_to_idx = checkpoint["class_to_idx"]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fake_probabilities = []
    frame_number = 0

    try:
        while len(fake_probabilities) < max_frames:
            success, frame = capture.read()
            if not success:
                break

            if frame_number % sample_every == 0:
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
                image = Image.fromarray(rgb_frame)

                result = predict_pil_image(
                    image,
                    model,
                    class_to_idx,
                    device,
                )
                fake_probabilities.append(
                    result.get("fake", 0.0)
                )

            frame_number += 1
    finally:
        capture.release()

    if not fake_probabilities:
        raise RuntimeError("No frames were sampled from the video.")

    mean_fake_probability = float(
        np.mean(fake_probabilities)
    )
    prediction = (
        "fake" if mean_fake_probability >= 0.5 else "real"
    )

    return {
        "prediction": prediction,
        "fake_probability": mean_fake_probability,
        "real_probability": 1.0 - mean_fake_probability,
        "sampled_frames": len(fake_probabilities),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a video is real or fake."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model-path",
        default="models/best_model.pth",
    )
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()

    result = predict_video(
        video_path=args.video,
        model_path=args.model_path,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
    )

    print(f"Prediction: {result['prediction'].upper()}")
    print(
        f"Fake probability: "
        f"{result['fake_probability']:.4f}"
    )
    print(
        f"Real probability: "
        f"{result['real_probability']:.4f}"
    )
    print(f"Sampled frames: {result['sampled_frames']}")


if __name__ == "__main__":
    main()
