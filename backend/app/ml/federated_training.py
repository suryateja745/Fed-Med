from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, random_split

from app.ml.dataset import MockMRISegmentationDataset
from app.ml.model import UNet3D
from app.ml.trainer import train_one_epoch, validate


def _make_loaders(
    batch_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
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

    return train_loader, val_loader


def initial_parameter_vector() -> np.ndarray:
    torch.manual_seed(42)

    model = UNet3D()

    vector = parameters_to_vector(
        model.parameters()
    )

    return (
        vector.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _model_from_vector(
    global_vector: np.ndarray,
) -> UNet3D:
    model = UNet3D()

    vector = torch.as_tensor(
        global_vector,
        dtype=torch.float32,
    )

    expected_size = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if vector.numel() != expected_size:
        raise ValueError(
            "Global parameter vector size mismatch: "
            f"received {vector.numel()}, "
            f"expected {expected_size}"
        )

    vector_to_parameters(
        vector,
        model.parameters(),
    )

    return model


def train_from_global_vector(
    global_vector: np.ndarray,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
) -> dict:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = _model_from_vector(
        global_vector
    ).to(device)

    train_loader, val_loader = _make_loaders(
        batch_size=batch_size
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history: list[dict[str, float]] = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_loss, dice = validate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "dice": float(dice),
            }
        )

    updated_vector = (
        parameters_to_vector(
            model.parameters()
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    final_metrics = history[-1]

    return {
        "updated_vector": updated_vector,
        "train_loss": final_metrics["train_loss"],
        "val_loss": final_metrics["val_loss"],
        "dice": final_metrics["dice"],
        "history": history,
    }


def evaluate_from_global_vector(
    global_vector: np.ndarray,
    batch_size: int = 1,
) -> tuple[float, float]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = _model_from_vector(
        global_vector
    ).to(device)

    _, val_loader = _make_loaders(
        batch_size=batch_size
    )

    return validate(
        model,
        val_loader,
        device,
    )