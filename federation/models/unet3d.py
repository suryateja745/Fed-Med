"""
3D U-Net Architecture and Model Utilities for Brain Tumor MRI Segmentation.

Supports multi-modal MRI scans (FLAIR, T1, T1ce, T2) and outputs multi-region
tumor segmentations (WT: Whole Tumor, TC: Tumor Core, ET: Enhancing Tumor).
Integrated with MONAI and PyTorch for local training and Flower parameter exchange.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try:
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet as MonaiUNet
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# Fallback Pure-PyTorch 3D U-Net (if MONAI is not yet installed)

class ConvBlock3D(nn.Module):
    """Dual 3D Convolution block with Normalization, LeakyReLU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_layer = (
            nn.BatchNorm3d if norm.lower() == "batch"
            else (nn.InstanceNorm3d if norm.lower() == "instance" else nn.Identity)
        )
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PyTorchUNet3D(nn.Module):

    # Standard 3D U-Net for volumetric medical image segmentation in PyTorch.

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        norm: str = "batch",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        # Encoders
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_in = in_channels
        for ch in channels[:-1]:
            self.encoders.append(ConvBlock3D(current_in, ch, norm=norm, dropout=dropout))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_in = ch

        # Bottleneck
        self.bottleneck = ConvBlock3D(channels[-2], channels[-1], norm=norm, dropout=dropout)

        # Decoders
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    rev_channels[i],
                    rev_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(
                ConvBlock3D(
                    rev_channels[i],
                    rev_channels[i + 1],
                    norm=norm,
                    dropout=dropout,
                )
            )

        # Final Classifier Head
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle potential odd dimension padding
            if x.shape != skip.shape:
                diff_d = skip.size(2) - x.size(2)
                diff_h = skip.size(3) - x.size(3)
                diff_w = skip.size(4) - x.size(4)
                x = nn.functional.pad(
                    x,
                    [
                        diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2,
                    ],
                )
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)


# FedMed 3D U-Net Model Wrapper (Prefers MONAI with PyTorch Fallback)

class FedMedUNet3D(nn.Module):
    """
    3D U-Net wrapper providing seamless integration with MONAI architectures,
    PyTorch fallback, and Flower federated weight serialization utilities.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = "batch",
        dropout: float = 0.2,
        use_monai: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.strides = tuple(strides)
        self.num_res_units = num_res_units
        self.norm = norm
        self.dropout = dropout

        if use_monai and HAS_MONAI:
            norm_layer = Norm.BATCH if norm.lower() == "batch" else Norm.INSTANCE
            self.backbone = MonaiUNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=self.channels,
                strides=self.strides,
                num_res_units=num_res_units,
                norm=norm_layer,
                dropout=dropout,
            )
            self.engine = "MONAI"
        else:
            self.backbone = PyTorchUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                channels=self.channels,
                norm=norm,
                dropout=dropout,
            )
            self.engine = "PyTorch-Native"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D U-Net."""
        return self.backbone(x)


# Model Factory & Initialization Utilities

def create_unet3d_model(
    in_channels: int = 4,
    out_channels: int = 3,
    channels: Optional[Tuple[int, ...]] = (16, 32, 64, 128, 256),
    strides: Optional[Tuple[int, ...]] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "batch",
    dropout: float = 0.2,
    init_weights_flag: bool = True,
) -> FedMedUNet3D:

    # Factory function to create and initialize a 3D U-Net model instance.
    model = FedMedUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels or (16, 32, 64, 128, 256),
        strides=strides or (2, 2, 2, 2),
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )
    if init_weights_flag:
        initialize_weights(model)
    return model


def build_unet3d_from_config(config: Optional[Dict[str, Any]] = None) -> FedMedUNet3D:
    """
    Build 3D U-Net model directly from a loaded configuration dictionary
    or from default fl_config.yaml.
    """
    if config is None:
        from federation.utils.config_loader import load_config
        config = load_config()

    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 4)
    out_channels = model_cfg.get("out_channels", 3)
    channels = tuple(model_cfg.get("channels", [16, 32, 64, 128, 256]))
    strides = tuple(model_cfg.get("strides", [2, 2, 2, 2]))
    num_res_units = model_cfg.get("num_res_units", 2)
    norm = model_cfg.get("norm", "batch")
    dropout = model_cfg.get("dropout", 0.2)

    return create_unet3d_model(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )


def initialize_weights(model: nn.Module, init_type: str = "kaiming") -> None:

    # Initialize convolutional, normalization, and linear layer weights.
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "non_trainable_parameters": total - trainable,
    }


# Parameter Extraction & Injection for Flower Federated Learning

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    # Extract model weights as a list of NumPy arrays (Flower format).
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    # Load a list of NumPy arrays (Flower aggregated weights) into model state_dict.
    current_state = model.state_dict()
    state_dict = {}
    for (key, orig_val), val in zip(current_state.items(), parameters):
        state_dict[key] = torch.as_tensor(val, dtype=orig_val.dtype, device=orig_val.device)
    model.load_state_dict(state_dict, strict=True)


# Model Checkpointing Utilities

def save_model_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    round_num: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    # Save model weights, optimizer state, round number, and metrics to disk.
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "round": round_num,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    return save_path


def load_model_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    # Load model weights and optional optimizer state from disk.
    
    load_path = Path(filepath)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {load_path}")

    checkpoint = torch.load(load_path, map_location=torch.device(device), weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        # Direct state dict
        model.load_state_dict(checkpoint, strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
