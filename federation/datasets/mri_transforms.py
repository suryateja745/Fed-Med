"""
MONAI Transforms and Preprocessing Pipeline for 3D Brain Tumor MRI Scans.
Handles multi-modal MRI scans (FLAIR, T1, T1ce, T2) and BraTS-style multi-region
segmentations (WT: Whole Tumor, TC: Tumor Core, ET: Enhancing Tumor).
"""

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
import torch

try:
    from monai.transforms import (
        Compose,
        EnsureTyped,
        MapTransform,
        NormalizeIntensityd,
        RandFlipd,
        RandGaussianNoised,
        RandSpatialCropd,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    MapTransform = object


# Helper File Loader
def load_file_to_array(path_or_array: Union[str, Path, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Load file path (.npy, .npz, .nii, .nii.gz) or return numpy array."""
    if isinstance(path_or_array, (str, Path)):
        p = Path(path_or_array)
        if p.suffix == ".npy":
            return np.load(p).astype(np.float32)
        elif p.suffix == ".npz":
            data = np.load(p)
            return data[list(data.keys())[0]].astype(np.float32)
        elif p.suffix in (".nii", ".gz"):
            try:
                import nibabel as nib
                return nib.load(str(p)).get_fdata(dtype=np.float32)
            except ImportError:
                raise ImportError(f"nibabel is required to load NIfTI file: {p}")
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")
    elif isinstance(path_or_array, torch.Tensor):
        return path_or_array.detach().cpu().numpy().astype(np.float32)
    elif isinstance(path_or_array, np.ndarray):
        return path_or_array.astype(np.float32)
    else:
        raise TypeError(f"Expected file path, ndarray, or Tensor, got {type(path_or_array).__name__}")


# Unified Loader Transform
class LoadImageOrArrayd(MapTransform if HAS_MONAI else object):
    """
    Unified loader transform for FedMed:
    Accepts filepaths (.nii, .nii.gz, .npy, .npz) as well as pre-loaded ndarrays/tensors.
    """

    def __init__(self, keys: Sequence[str] = ("image", "label")) -> None:
        if HAS_MONAI:
            super().__init__(keys)
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = load_file_to_array(d[key])
        return d


class Ensure4DChannelFirstd(MapTransform if HAS_MONAI else object):
    """
    Ensure volumetric data has shape [Channels, D, H, W].
    If 3D [D, H, W], unsqueezes channel dimension to [1, D, H, W].
    If 4D [C, D, H, W], preserves existing channels.
    """

    def __init__(self, keys: Sequence[str] = ("image", "label")) -> None:
        if HAS_MONAI:
            super().__init__(keys)
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            val = d[key]
            if isinstance(val, np.ndarray):
                if val.ndim == 3:
                    val = np.expand_dims(val, axis=0)
            elif isinstance(val, torch.Tensor):
                if val.ndim == 3:
                    val = val.unsqueeze(0)
            d[key] = val
        return d


# BraTS Label Conversion Transform
class ConvertToMultiChannelBasedOnBratsClassesd:
    """
    Convert BraTS segmentation labels to 3 multi-region binary channels:
    - Channel 0: TC (Tumor Core: label 1 + label 4)
    - Channel 1: WT (Whole Tumor: label 1 + label 2 + label 4)
    - Channel 2: ET (Enhancing Tumor: label 4)
    """

    def __init__(self, keys: Sequence[str] = ("label",)) -> None:
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            seg = d[key]
            if isinstance(seg, (str, Path)):
                seg = load_file_to_array(seg)

            if isinstance(seg, torch.Tensor):
                tc = torch.logical_or(seg == 1, seg == 4)
                wt = torch.logical_or(tc, seg == 2)
                et = seg == 4
                d[key] = torch.stack([tc, wt, et], dim=0).float()
            elif isinstance(seg, np.ndarray):
                tc = np.logical_or(seg == 1, seg == 4)
                wt = np.logical_or(tc, seg == 2)
                et = seg == 4
                d[key] = np.stack([tc, wt, et], axis=0).astype(np.float32)
        return d


# Pure-PyTorch / NumPy Fallback Transforms
class SimpleDictTransform:
    """Fallback transform composing simple numpy/torch tensor operations."""

    def __init__(
        self,
        roi_size: Tuple[int, int, int] = (64, 64, 64),
        is_train: bool = True,
        convert_brats: bool = False,
    ) -> None:
        self.roi_size = roi_size
        self.is_train = is_train
        self.convert_brats = convert_brats
        self.brats_converter = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])

    def __call__(self, sample: dict) -> dict:
        data = dict(sample)
        image = data["image"]
        label = data["label"]

        if isinstance(image, (str, Path)):
            image = load_file_to_array(image)
        if isinstance(label, (str, Path)):
            label = load_file_to_array(label)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)

        # Ensure channel first: [C, D, H, W]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if label.ndim == 3:
            label = label.unsqueeze(0)

        data["image"] = image
        data["label"] = label

        if self.convert_brats:
            data = self.brats_converter(data)
            label = data["label"]

        # Intensity Normalization (zero mean, unit variance per channel)
        for c in range(image.shape[0]):
            c_mean = image[c].mean()
            c_std = image[c].std()
            if c_std > 1e-6:
                image[c] = (image[c] - c_mean) / c_std

        # Spatial Crop or Pad to roi_size
        image = self._crop_or_pad(image, self.roi_size)
        label = self._crop_or_pad(label, self.roi_size)

        # Data augmentation for training
        if self.is_train and np.random.rand() > 0.5:
            axis = np.random.choice([1, 2, 3])
            image = torch.flip(image, dims=[axis])
            label = torch.flip(label, dims=[axis])

        data["image"] = image
        data["label"] = label
        return data

    @staticmethod
    def _crop_or_pad(tensor: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        c, d, h, w = tensor.shape
        td, th, tw = target_size

        if d > td:
            sd = (d - td) // 2
            tensor = tensor[:, sd : sd + td, :, :]
        elif d < td:
            pad_d = td - d
            tensor = torch.nn.functional.pad(tensor, [0, 0, 0, 0, pad_d // 2, pad_d - pad_d // 2])

        if h > th:
            sh = (h - th) // 2
            tensor = tensor[:, :, sh : sh + th, :]
        elif h < th:
            pad_h = th - h
            tensor = torch.nn.functional.pad(tensor, [0, 0, pad_h // 2, pad_h - pad_h // 2, 0, 0])

        if w > tw:
            sw = (w - tw) // 2
            tensor = tensor[:, :, :, sw : sw + tw]
        elif w < tw:
            pad_w = tw - w
            tensor = torch.nn.functional.pad(tensor, [pad_w // 2, pad_w - pad_w // 2, 0, 0, 0, 0])

        return tensor


# MONAI Transform Pipelines
def get_train_transforms(
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    convert_brats: bool = False,
):
    """
    Build MONAI preprocessing and augmentation pipeline for training.
    """
    if not HAS_MONAI:
        return SimpleDictTransform(roi_size=roi_size, is_train=True, convert_brats=convert_brats)

    transforms_list = [
        LoadImageOrArrayd(keys=["image", "label"]),
        Ensure4DChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    if convert_brats:
        transforms_list.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]))

    transforms_list.extend([
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        EnsureTyped(keys=["image", "label"], data_type="tensor"),
    ])

    return Compose(transforms_list)


def get_val_transforms(
    roi_size: Optional[Tuple[int, int, int]] = None,
    convert_brats: bool = False,
):
    """
    Build MONAI preprocessing pipeline for validation/evaluation.
    """
    if not HAS_MONAI:
        return SimpleDictTransform(roi_size=roi_size or (64, 64, 64), is_train=False, convert_brats=convert_brats)

    transforms_list = [
        LoadImageOrArrayd(keys=["image", "label"]),
        Ensure4DChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    if convert_brats:
        transforms_list.append(ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]))

    if roi_size is not None:
        transforms_list.append(RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False))

    transforms_list.append(EnsureTyped(keys=["image", "label"], data_type="tensor"))

    return Compose(transforms_list)
