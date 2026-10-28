import argparse

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    fake_index: int,
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
        # F1 for the FAKE class, looked up from the class mapping
        # rather than assuming fake == 0.
        "f1": f1_score(
            labels_all,
            predictions_all,
            average="binary",
            pos_label=fake_index,
            zero_division=0,
        ),
    }


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """
    Step 2: weight each class inversely to how often it appears,
    so the loss does not favour the majority class.
    weight[c] = total / (num_classes * count[c])
    """
    counts = torch.bincount(
        torch.tensor(targets), minlength=num_classes
    ).float()

    if (counts == 0).any():
        raise ValueError(f"A class has no training images: {counts.tolist()}")

    return counts.sum() / (num_classes * counts)


def main():
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet deepfake image classifier."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model-output",
        default="models/best_model_faces.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting in the loss.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Stop after this many epochs without validation-loss "
             "improvement.",
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
    fake_index = class_to_idx["fake"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    print(f"Class mapping: {class_to_idx}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # --- Step 2: class balance -------------------------------------
    train_targets = train_loader.dataset.targets
    for index in sorted(idx_to_class):
        print(
            f"  train {idx_to_class[index]}: "
            f"{train_targets.count(index)} images"
        )

    if args.no_class_weights:
        class_weights = None
        print("Class weights: disabled")
    else:
        class_weights = compute_class_weights(
            train_targets, num_classes=len(class_to_idx)
        )
        print(
            "Class weights: "
            + ", ".join(
                f"{idx_to_class[i]}={w:.3f}"
                for i, w in enumerate(class_weights.tolist())
            )
        )
        class_weights = class_weights.to(device)

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
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Step 3: scheduler + early stopping ------------------------
    # Halve the learning rate when validation loss stops improving.
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    ensure_parent(args.model_output)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            fake_index=fake_index,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            fake_index=fake_index,
            optimizer=None,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.1e} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

        # Select the best model on validation LOSS: it is smoother
        # than F1 at a fixed 0.5 threshold, and the decision
        # threshold is calibrated separately afterwards.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": 224,
                    "best_validation_loss": best_val_loss,
                    "best_validation_f1": val_metrics["f1"],
                    "best_validation_accuracy": val_metrics["accuracy"],
                    "epoch": epoch,
                    "class_weights": (
                        None if class_weights is None
                        else class_weights.cpu().tolist()
                    ),
                },
                args.model_output,
            )
            print(f"Saved improved model to {args.model_output}")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: no validation-loss improvement "
                    f"for {args.patience} epochs."
                )
                break

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
