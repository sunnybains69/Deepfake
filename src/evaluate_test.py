
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .dataset import get_transforms
from .model import load_model
from .utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/faceforensics_prepared"
    )
    parser.add_argument(
        "--model-path",
        default="models/best_model_ffpp.pth"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()

    test_dir = Path(args.data_dir) / "test"
    test_dataset = ImageFolder(
        root=test_dir,
        transform=get_transforms(train=False),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model, checkpoint = load_model(args.model_path, device)
    class_to_idx = checkpoint["class_to_idx"]

    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Test dataset class mapping does not match model."
        )

    model.eval()

    labels_all = []
    predictions_all = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())

    fake_index = class_to_idx["fake"]
    labels_order = sorted(class_to_idx.values())
    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }

    print(f"Test images: {len(test_dataset)}")
    print(
        f"Accuracy: "
        f"{accuracy_score(labels_all, predictions_all):.4f}"
    )
    print(
        f"Precision: "
        f"{precision_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )
    print(
        f"Recall: "
        f"{recall_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )
    print(
        f"F1 score: "
        f"{f1_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            labels_all,
            predictions_all,
            labels=labels_order,
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            labels_all,
            predictions_all,
            labels=labels_order,
            target_names=[idx_to_class[i] for i in labels_order],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
