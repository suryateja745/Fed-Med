"""
Loss Functions and Evaluation Metrics for 3D Brain Tumor MRI Segmentation.
Provides DiceLoss, DiceCELoss, and per-region Dice metric computations
(Whole Tumor, Tumor Core, Enhancing Tumor) with MONAI and native PyTorch support.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.losses import DiceCELoss as MonaiDiceCELoss
    from monai.losses import DiceLoss as MonaiDiceLoss
    from monai.metrics import DiceMetric as MonaiDiceMetric
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# Native PyTorch Loss Implementations (Fallback & Standalone)

class PyTorchDiceLoss(nn.Module):
    """
    Multi-channel Soft Dice Loss for volumetric binary/multi-label segmentation.
    """

    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten spatial dimensions: [B, C, D*H*W]
        pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)

        intersection = 2.0 * torch.sum(pred_flat * target_flat, dim=-1) + self.smooth
        denominator = torch.sum(pred_flat, dim=-1) + torch.sum(target_flat, dim=-1) + self.smooth

        dice_score = intersection / denominator
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss


class PyTorchDiceCELoss(nn.Module):
    """
    Combined Soft Dice Loss + Binary Cross-Entropy (with Logits) Loss.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = PyTorchDiceLoss(smooth=smooth, sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice_loss(pred, target)
        loss_bce = self.bce_loss(pred, target)
        return self.dice_weight * loss_dice + self.ce_weight * loss_bce


# Loss Factory Function

def get_loss_function(
    loss_name: str = "DiceCELoss",
    sigmoid: bool = True,
    use_monai: bool = True,
) -> nn.Module:
    """
    Build loss function instance. Supports 'DiceCELoss', 'DiceLoss', 'BCEWithLogitsLoss'.
    """
    name = loss_name.lower()
    if use_monai and HAS_MONAI:
        if "dicece" in name:
            return MonaiDiceCELoss(sigmoid=sigmoid, smooth_nr=1e-5, smooth_dr=1e-5)
        elif "dice" in name:
            return MonaiDiceLoss(sigmoid=sigmoid, smooth_nr=1e-5, smooth_dr=1e-5)

    # Native PyTorch fallback
    if "dicece" in name:
        return PyTorchDiceCELoss()
    elif "dice" in name:
        return PyTorchDiceLoss(sigmoid=sigmoid)
    elif "bce" in name:
        return nn.BCEWithLogitsLoss()
    else:
        return PyTorchDiceCELoss()


# Dice Score Metric Computation (Per-Region & Mean)

def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
    apply_sigmoid: bool = True,
) -> Dict[str, float]:
    """
    Calculate Dice Similarity Coefficient (DSC) for multi-channel segmentations.

    Args:
        pred: Predicted logits [B, C, D, H, W] or binarized predictions.
        target: Ground-truth binary masks [B, C, D, H, W].
        threshold: Binarization threshold.
        smooth: Epsilon to prevent division by zero.
        apply_sigmoid: Apply sigmoid activation to logits if True.

    Returns:
        Dictionary containing:
        - 'dice_mean': Overall average Dice score across all channels.
        - 'dice_ch_{i}': Dice score for channel i (e.g. TC, WT, ET).
    """
    if apply_sigmoid:
        pred = torch.sigmoid(pred)

    pred_bin = (pred >= threshold).float()
    target = target.float()

    num_channels = pred.shape[1]
    channel_names = ["tc", "wt", "et"]  # Standard BraTS naming for 3 channels

    results = {}
    channel_scores = []

    for c in range(num_channels):
        p_c = pred_bin[:, c].contiguous().view(-1)
        t_c = target[:, c].contiguous().view(-1)

        intersection = 2.0 * torch.sum(p_c * t_c).item() + smooth
        denominator = torch.sum(p_c).item() + torch.sum(t_c).item() + smooth
        score = float(intersection / denominator)

        # Record per-channel score
        name = channel_names[c] if c < len(channel_names) else f"ch_{c}"
        results[f"dice_{name}"] = round(score, 4)
        channel_scores.append(score)

    results["dice_mean"] = round(float(np.mean(channel_scores)), 4)
    return results
