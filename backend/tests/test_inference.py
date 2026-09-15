from pathlib import Path

import torch

from app.ml.dataset import MockMRISegmentationDataset
from app.ml.inference import segment_volume


def test_segmentation_from_checkpoint(tmp_path: Path) -> None:
    from app.ml.trainer import train_local_model

    result = train_local_model(
        epochs=1,
        batch_size=1,
        checkpoint_dir=tmp_path,
    )

    dataset = MockMRISegmentationDataset(size=1)
    image, mask = dataset[0]

    output = segment_volume(
        image,
        result["best_checkpoint"],
    )

    predicted = output["mask"]

    assert tuple(predicted.shape) == (
        1,
        32,
        32,
        32,
    )

    assert set(
        torch.unique(predicted).tolist()
    ).issubset({0, 1})