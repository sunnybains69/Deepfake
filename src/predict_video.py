
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
    model_path: str = "models/best_model_faces.pth",
    sample_every: int = 15,
    max_frames: int = 60,
    crop_faces: bool = True,
):
    # Check video path
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    if sample_every < 1 or max_frames < 1:
        raise ValueError(
            "sample_every and max_frames must be at least 1."
        )

    # Load trained EfficientNet-B0 model
    device = get_device()

    model, checkpoint = load_model(
        model_path,
        device,
    )

    model.eval()

    class_to_idx = checkpoint["class_to_idx"]

    # Open video
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    fake_probabilities = []
    frame_number = 0
    skipped_faces = 0

    # Analyse video frames
    try:
        while len(fake_probabilities) < max_frames:

            success, frame = capture.read()

            if not success:
                break

            if frame_number % sample_every == 0:

                # Convert OpenCV BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                image = Image.fromarray(rgb_frame)

                # Detect and analyse face
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
                        and str(error)
                        == "No face detected in the image."
                    ):
                        skipped_faces += 1
                        frame_number += 1
                        continue

                    raise

                # Record fake probability
                fake_probabilities.append(
                    result["fake"]
                )

                # Display individual frame results
                print(
                    f"Frame {frame_number}: "
                    f"FAKE={result['fake']:.2%}, "
                    f"REAL={result['real']:.2%}"
                )

            frame_number += 1

    finally:
        capture.release()

    # Check that usable frames were found
    if not fake_probabilities:
        raise RuntimeError(
            "No usable frames were found in the video."
        )

    # Calculate average fake probability
    mean_fake_probability = float(
        np.mean(fake_probabilities)
    )

    mean_real_probability = (
        1.0 - mean_fake_probability
    )

    # Classification with uncertainty range
    # Thresholds are provisional, not calibrated
    if 0.45 <= mean_fake_probability <= 0.55:
        prediction = "uncertain"

    elif mean_fake_probability > 0.55:
        prediction = "fake"

    else:
        prediction = "real"

    # Display summary in terminal
    print("\nVideo analysis completed")
    print(f"Prediction: {prediction.upper()}")
    print(
        f"Fake probability: "
        f"{mean_fake_probability:.2%}"
    )
    print(
        f"Real probability: "
        f"{mean_real_probability:.2%}"
    )
    print(
        f"Frames analysed: "
        f"{len(fake_probabilities)}"
    )
    print(
        f"Frames skipped (no face detected): "
        f"{skipped_faces}"
    )

    # Return results to Streamlit
    return {
        "prediction": prediction,
        "fake_probability": mean_fake_probability,
        "real_probability": mean_real_probability,
        "sampled_frames": len(fake_probabilities),
        "skipped_faces": skipped_faces,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict whether a video is real, "
            "fake, or uncertain."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video file.",
    )

    parser.add_argument(
        "--model-path",
        default="models/best_model_faces.pth",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--no-crop-faces",
        action="store_true",
        help="Disable face detection and cropping.",
    )

    args = parser.parse_args()

    predict_video(
        video_path=args.video,
        model_path=args.model_path,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        crop_faces=not args.no_crop_faces,
    )


if __name__ == "__main__":
    main()
