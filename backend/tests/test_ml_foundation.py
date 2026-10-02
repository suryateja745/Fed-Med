import torch

from app.ml.config import TrainingConfig
from app.ml.dataset import MockMRISegmentationDataset
from app.ml.model import UNet3D


def test_unet3d_forward_shape() -> None:
    model = UNet3D()
    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 2, 32, 32, 32)


def test_mock_mri_dataset_shape_and_classes() -> None:
    dataset = MockMRISegmentationDataset()
    image, mask = dataset[0]

    assert tuple(image.shape) == (1, 32, 32, 32)
    assert tuple(mask.shape) == (32, 32, 32)
    assert set(mask.unique().tolist()).issubset({0, 1})


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.epochs == 2
    assert config.batch_size == 1
    assert config.learning_rate == 0.001
    assert config.selected_hospitals == (
        "hospital-1",
        "hospital-2",
        "hospital-3",
    )