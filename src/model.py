import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def build_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(input_features, 2),
    )

    return model


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
