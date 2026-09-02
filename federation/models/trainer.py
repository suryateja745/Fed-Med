"""
Local Training and Validation Utilities for FedMed Flower Clients.
Manages local training epochs, gradient clipping, learning rate scheduling,
optimizer setups, and validation metric evaluations on hospital nodes.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from federation.models.metrics import compute_dice_score, get_loss_function


# Optimizer & Scheduler Factories

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "AdamW",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """
    Construct optimizer for local model training.
    """
    name = optimizer_name.lower()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if "adamw" in name:
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif "adam" in name:
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif "sgd" in name:
        return torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 10,
    eta_min: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Construct learning rate scheduler.
    """
    name = (scheduler_name or "").lower()
    if "cosine" in name:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)
    elif "step" in name:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


# Single Epoch Training & Validation Loops

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = "cpu",
    gradient_clip_val: float = 1.0,
) -> Dict[str, float]:
    """
    Execute a single training epoch across all batches in train_loader.

    Returns:
        Dictionary with 'train_loss' and 'num_samples'.
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        images = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.float32)
        batch_size = images.shape[0]

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if gradient_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    return {
        "train_loss": round(float(avg_loss), 4),
        "num_samples": total_samples,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model performance and calculate Dice similarity metrics.

    Returns:
        Dictionary with 'val_loss', 'val_dice', and per-region Dice scores.
    """
    model.eval()
    model.to(device)

    if loss_fn is None:
        loss_fn = get_loss_function("DiceCELoss")

    total_loss = 0.0
    total_samples = 0
    all_dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.float32)
            batch_size = images.shape[0]

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute Dice scores for batch
            dice_metrics = compute_dice_score(outputs, labels)
            all_dice_scores.append(dice_metrics)

    avg_loss = total_loss / max(1, total_samples)

    # Average metrics across batches
    results = {"val_loss": round(float(avg_loss), 4)}
    if all_dice_scores:
        for key in all_dice_scores[0].keys():
            mean_metric = float(np.mean([d[key] for d in all_dice_scores]))
            results[f"val_{key}"] = round(mean_metric, 4)
    else:
        results["val_dice_mean"] = 0.0

    return results


# Multi-Epoch Local Client Training Routine (Flower Integration)

def train_local_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    optimizer_name: str = "AdamW",
    scheduler_name: str = "cosine",
    loss_name: str = "DiceCELoss",
    device: str = "cpu",
    gradient_clip_val: float = 1.0,
) -> Dict[str, Any]:
    """
    Full local training pipeline executed on a hospital client during a federated round.

    Returns:
        Consolidated metrics dictionary reporting training loss, validation Dice,
        and sample counts for Flower server reporting.
    """
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, scheduler_name=scheduler_name, epochs=epochs)
    loss_fn = get_loss_function(loss_name)

    metrics_history = []
    total_trained_samples = 0

    for epoch in range(1, epochs + 1):
        epoch_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            gradient_clip_val=gradient_clip_val,
        )
        total_trained_samples = epoch_metrics["num_samples"]
        metrics_history.append(epoch_metrics["train_loss"])

        if scheduler is not None:
            scheduler.step()

    final_train_loss = metrics_history[-1] if metrics_history else 0.0

    results = {
        "train_loss": round(float(final_train_loss), 4),
        "num_samples": total_trained_samples,
        "epochs_completed": epochs,
    }

    # Optional local validation evaluation
    if val_loader is not None and len(val_loader) > 0:
        val_metrics = validate(model=model, val_loader=val_loader, loss_fn=loss_fn, device=device)
        results.update(val_metrics)

    return results
