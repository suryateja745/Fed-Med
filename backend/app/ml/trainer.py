from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from app.ml.dataset import MockMRISegmentationDataset
from app.ml.model import UNet3D


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate foreground Dice score for binary segmentation.
    """
    predicted_labels = torch.argmax(predictions, dim=1)
    predicted_fg = (predicted_labels == 1).float()
    target_fg = (targets == 1).float()

    intersection = (predicted_fg * target_fg).sum()
    denominator = predicted_fg.sum() + target_fg.sum()

    dice = (2.0 * intersection + epsilon) / (
        denominator + epsilon
    )

    return float(dice.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += float(loss.item())
        total_dice += dice_score(outputs, masks)
        batches += 1

    return (
        total_loss / max(batches, 1),
        total_dice / max(batches, 1),
    )


def train_local_model(
    epochs: int = 2,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "data/checkpoints",
) -> dict[str, Any]:
    """
    Run a small local 3D U-Net training session.

    Uses the synthetic MRI-style dataset currently available in
    the development environment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MockMRISegmentationDataset(size=8)

    train_size = max(1, int(len(dataset) * 0.75))
    val_size = len(dataset) - train_size

    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = UNet3D().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[dict[str, float]] = []
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, val_dice = validate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice": val_dice,
            }
        )

        latest_path = checkpoint_path / "latest_model.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dice": val_dice,
            },
            latest_path,
        )

        if val_dice > best_dice:
            best_dice = val_dice

            best_path = checkpoint_path / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "dice": val_dice,
                },
                best_path,
            )

    return {
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": history,
        "best_dice": best_dice,
        "latest_checkpoint": str(
            checkpoint_path / "latest_model.pt"
        ),
        "best_checkpoint": str(
            checkpoint_path / "best_model.pt"
        ),
    }