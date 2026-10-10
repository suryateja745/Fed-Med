from __future__ import annotations

from pathlib import Path

import torch

from app.ml.model import UNet3D
from app.ml.trainer import dice_score


def load_model(checkpoint_path: str | Path) -> UNet3D:
    model = UNet3D()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def segment_volume(
    volume: torch.Tensor,
    checkpoint_path: str | Path,
) -> dict:
    model = load_model(checkpoint_path)

    if volume.ndim == 4:
        volume = volume.unsqueeze(0)

    with torch.no_grad():
        logits = model(volume)
        mask = torch.argmax(logits, dim=1)

    return {
        "mask": mask,
        "logits": logits,
    }


def calculate_segmentation_dice(
    predicted_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    predicted = predicted_mask.unsqueeze(1).float()
    target = target_mask.unsqueeze(1).long()

    logits = torch.cat(
        [
            (1.0 - predicted).unsqueeze(1),
            predicted.unsqueeze(1),
        ],
        dim=1,
    )

    return dice_score(logits, target)