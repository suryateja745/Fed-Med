from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class MockMRISegmentationDataset(Dataset):
    """
    Small synthetic 3D MRI-style dataset.

    This is the development fallback when a real MRI dataset
    is not available locally.
    """

    def __init__(
        self,
        size: int = 8,
        volume_size: tuple[int, int, int] = (32, 32, 32),
    ) -> None:
        self.size = size
        self.volume_size = volume_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ):
        torch.manual_seed(index)

        image = torch.randn(
            1,
            *self.volume_size,
        )

        mask = torch.zeros(
            *self.volume_size,
            dtype=torch.long,
        )

        center = tuple(
            dimension // 2
            for dimension in self.volume_size
        )

        radius = min(
            self.volume_size
        ) // 6

        zz, yy, xx = torch.meshgrid(
            torch.arange(
                self.volume_size[0]
            ),
            torch.arange(
                self.volume_size[1]
            ),
            torch.arange(
                self.volume_size[2]
            ),
            indexing="ij",
        )

        distance = (
            (zz - center[0]) ** 2
            + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2
        )

        mask[
            distance < radius**2
        ] = 1

        return image.float(), mask