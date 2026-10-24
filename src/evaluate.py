import argparse

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .dataset import create_dataloaders
from .model import load_model
from .utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained deepfake classifier."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--model-path",
        default="models/best_model.pth",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    _, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, checkpoint = load_model(args.model_path, device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        index: name for name, index in class_to_idx.items()
    }

    labels_all = []
    predictions_all = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())

    fake_index = class_to_idx["fake"]

    print(f"Accuracy:  {accuracy_score(labels_all, predictions_all):.4f}")
    print(
        "Precision: "
        f"{precision_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )
    print(
        "F1 score:  "
        f"{f1_score(labels_all, predictions_all, pos_label=fake_index, zero_division=0):.4f}"
    )

    labels_order = sorted(idx_to_class)
    target_names = [idx_to_class[index] for index in labels_order]

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
            target_names=target_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
