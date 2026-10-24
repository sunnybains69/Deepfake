import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from .dataset import create_dataloaders
from .model import build_model
from .utils import ensure_parent, get_device, set_seed


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    labels_all = []
    predictions_all = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    dataset_size = len(loader.dataset)

    return {
        "loss": running_loss / dataset_size,
        "accuracy": accuracy_score(labels_all, predictions_all),
        "f1": f1_score(
            labels_all,
            predictions_all,
            average="binary",
            pos_label=0,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet deepfake image classifier."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model-output",
        default="models/best_model.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Class mapping: {class_to_idx}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    model = build_model(
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    ensure_parent(args.model_output)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": 224,
                    "best_validation_f1": best_f1,
                    "epoch": epoch,
                },
                args.model_output,
            )
            print(
                f"Saved improved model to {args.model_output}"
            )

    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
