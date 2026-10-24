from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
):
    root = Path(data_dir)

    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected data/train and data/val directories."
        )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_transforms(train=False),
    )

    expected_classes = {"fake", "real"}
    if set(train_dataset.classes) != expected_classes:
        raise ValueError(
            f"Expected train classes {sorted(expected_classes)}, "
            f"found {train_dataset.classes}."
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings are different."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.class_to_idx
