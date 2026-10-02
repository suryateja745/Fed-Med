from __future__ import annotations

import torch
from torch import nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """MONAI-based compact 3D U-Net for FedMed brain-tumor segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
    ) -> None:
        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(8, 16, 32),
            strides=(2, 2),
            num_res_units=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)