import argparse
from pathlib import Path

import torch
from PIL import Image

from .dataset import get_transforms
from .model import load_model
from .utils import get_device


def predict_pil_image(
    image: Image.Image,
    model,
    class_to_idx: dict,
    device: torch.device,
) -> dict:
    transform = get_transforms(train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    idx_to_class = {
        index: name for name, index in class_to_idx.items()
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
