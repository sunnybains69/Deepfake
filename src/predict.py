import cv2
import numpy as np
import argparse
from pathlib import Path

import torch
from PIL import Image

from .dataset import get_transforms
from .model import load_model
from .utils import get_device


def crop_face(image: Image.Image) -> Image.Image | None:
    """
    Detect and crop the largest face using the same
    settings as prepare_dataset_faces.py.
    """

    rgb = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Could not load OpenCV face detector."
        )

    faces = detector.detectMultiScale(
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

    # Same 25% margin used during training
    margin = int(max(w, h) * 0.25)

    height, width = rgb.shape[:2]

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)

    face = rgb[y1:y2, x1:x2]

    if face.size == 0:
        return None

    # Resize using the same OpenCV interpolation
    face = cv2.resize(
        face,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return Image.fromarray(face)

def predict_pil_image(
    image: Image.Image,
    model,
    class_to_idx: dict,
    device: torch.device,
    crop_faces: bool = False,
) -> dict:

    if crop_faces:
        cropped = crop_face(image)

        if cropped is None:
            raise ValueError(
                "No face detected in the image."
            )

        image = cropped

    transform = get_transforms(train=False)

    tensor = (
        transform(image.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(
            logits, dim=1
        )[0].cpu()

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    result = {
        idx_to_class[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }

    predicted_index = int(probabilities.argmax().item())

    result["prediction"] = idx_to_class[predicted_index]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether one image is real or fake."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model-path",
        default="models/best_model.pth",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    model, checkpoint = load_model(args.model_path, device)

    with Image.open(image_path) as image:
        result = predict_pil_image(
            image,
            model,
            checkpoint["class_to_idx"],
            device,
        )

    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Fake probability: {result.get('fake', 0.0):.4f}")
    print(f"Real probability: {result.get('real', 0.0):.4f}")


if __name__ == "__main__":
    main()
