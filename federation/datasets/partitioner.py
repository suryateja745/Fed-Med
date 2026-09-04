"""
Hospital Dataset Partitioner and Local Ingestion Pipeline.
Manages local MRI data loading for individual hospital nodes (Hospital A, B, C)
without uploading or transmitting raw medical scans.
Supports IID, Non-IID Dirichlet, and Quantity-Skew partitioning schemes.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from federation.datasets.mri_transforms import (
    SimpleDictTransform,
    get_train_transforms,
    get_val_transforms,
)


# Local MRI Dataset Class
class FedMedMRIDataset(Dataset):
    """
    Volumetric 3D MRI Dataset for local hospital nodes.
    Supports both file path dictionaries and preloaded arrays/tensors.
    """

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transform: Optional[Any] = None,
    ) -> None:
        self.data_list = data_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        data = dict(item)

        # Apply Transforms
        if self.transform is not None:
            data = self.transform(data)
        else:
            # Direct load fallback if no transform pipeline
            if isinstance(data.get("image"), (str, Path)):
                data["image"] = self._load_file(data["image"])
            if isinstance(data.get("label"), (str, Path)):
                data["label"] = self._load_file(data["label"])

        # Final tensor validation
        if not isinstance(data["image"], torch.Tensor):
            data["image"] = torch.tensor(data["image"], dtype=torch.float32)
        if not isinstance(data["label"], torch.Tensor):
            data["label"] = torch.tensor(data["label"], dtype=torch.float32)

        return data

    @staticmethod
    def _load_file(path: Union[str, Path]) -> np.ndarray:
        p = Path(path)
        if p.suffix == ".npy":
            return np.load(p).astype(np.float32)
        elif p.suffix == ".npz":
            data = np.load(p)
            return data[list(data.keys())[0]].astype(np.float32)
        elif p.suffix in (".nii", ".gz"):
            try:
                import nibabel as nib
                nimg = nib.load(str(p))
                return nimg.get_fdata(dtype=np.float32)
            except ImportError:
                raise ImportError(f"nibabel is required to load NIfTI file: {p}")
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")


def _extract_base_identifier(path: Path) -> str:
    """Strip file extensions and modality/segmentation suffixes to extract the case ID."""
    name = path.name.lower()
    for ext in [".nii.gz", ".nii", ".npy", ".npz", ".h5"]:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    for suffix in ["_image", "_img", "_scan", "_raw", "_seg", "_label", "_mask", "_annotation"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


# Local Directory Scanner
def scan_local_dataset(data_dir: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Recursively scan a local hospital directory for paired MRI scans and segmentation masks.
    Supports formats: .nii, .nii.gz, .npy, .npz
    """
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Local dataset directory does not exist: {directory}")

    # Check for manifest dataset.json (e.g. Medical Segmentation Decathlon format)
    manifest_path = directory / "dataset.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            training_items = manifest.get("training", [])
            data_list = []
            for item in training_items:
                img_path = str(directory / item["image"])
                lbl_path = str(directory / item["label"])
                data_list.append({"image": img_path, "label": lbl_path})
            if data_list:
                return data_list

    # Separate subfolder pattern: imagesTr/ & labelsTr/
    images_dir = directory / "imagesTr"
    labels_dir = directory / "labelsTr"
    if images_dir.exists() and labels_dir.exists():
        img_files = sorted(list(images_dir.glob("*.nii*")) + list(images_dir.glob("*.npy")) + list(images_dir.glob("*.npz")))
        data_list = []
        for img_p in img_files:
            lbl_p = labels_dir / img_p.name
            if lbl_p.exists():
                data_list.append({"image": str(img_p), "label": str(lbl_p)})
        if data_list:
            return data_list

    # Generic file pairing by base ID matching
    all_files = sorted(list(directory.rglob("*.nii*")) + list(directory.rglob("*.npy")) + list(directory.rglob("*.npz")))
    image_files = [
        f for f in all_files
        if not any(k in f.name.lower() for k in ["seg", "label", "mask", "annotation"])
    ]
    label_files = [
        f for f in all_files
        if any(k in f.name.lower() for k in ["seg", "label", "mask", "annotation"])
    ]

    label_map = {_extract_base_identifier(lbl): lbl for lbl in label_files}

    data_list = []
    for img_p in image_files:
        base_id = _extract_base_identifier(img_p)
        if base_id in label_map:
            data_list.append({"image": str(img_p), "label": str(label_map[base_id])})

    return data_list


# Federated Dataset Partitioner (IID & Non-IID)
def partition_dataset(
    data_list: List[Dict[str, Any]],
    num_clients: int = 3,
    partition_type: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partition dataset across hospital clients.

    Args:
        data_list: Full list of sample dictionaries.
        num_clients: Number of hospital partitions (default 3: Hospital A, B, C).
        partition_type: 'iid', 'quantity_skew', or 'dirichlet'.
        alpha: Dirichlet concentration parameter for non-IID distributions.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping hospital IDs to their allocated sample lists.
    """
    np.random.seed(seed)
    n_samples = len(data_list)
    indices = np.random.permutation(n_samples)
    hospital_names = [f"hospital_{chr(97 + i)}" for i in range(num_clients)]  # hospital_a, hospital_b, hospital_c

    partitions: Dict[str, List[Dict[str, Any]]] = {h: [] for h in hospital_names}

    if n_samples == 0:
        return partitions

    if partition_type.lower() == "iid":
        splits = np.array_split(indices, num_clients)
        for i, h_name in enumerate(hospital_names):
            partitions[h_name] = [data_list[idx] for idx in splits[i]]

    elif partition_type.lower() == "quantity_skew":
        # Unequal dataset size distribution (e.g. 50%, 30%, 20%)
        proportions = np.array([0.5, 0.3, 0.2][:num_clients])
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * n_samples).astype(int)
        splits = np.split(indices, split_points[:-1])
        for i, h_name in enumerate(hospital_names):
            partitions[h_name] = [data_list[idx] for idx in splits[i]]

    elif partition_type.lower() == "dirichlet":
        # Dirichlet Non-IID partition
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * n_samples).astype(int)
        splits = np.split(indices, split_points[:-1])
        for i, h_name in enumerate(hospital_names):
            partitions[h_name] = [data_list[idx] for idx in splits[i]]

    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")

    return partitions


# Synthetic MRI Generator for Testing & Zero-Data Bootstrap
def generate_synthetic_mri_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 6,
    spatial_size: Tuple[int, int, int] = (32, 32, 32),
    in_channels: int = 4,
    out_channels: int = 3,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Generate synthetic 3D MRI volumes and multi-region tumor masks for local testing.
    """
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_list = []
    d, h, w = spatial_size

    for i in range(num_samples):
        # Generate synthetic 4-channel MRI (T1, T1ce, T2, FLAIR)
        image = np.random.randn(in_channels, d, h, w).astype(np.float32)

        # Generate synthetic 3-region tumor mask (TC, WT, ET)
        # Create a sphere tumor region in the center
        label = np.zeros((out_channels, d, h, w), dtype=np.float32)
        center = (d // 2 + np.random.randint(-3, 4), h // 2 + np.random.randint(-3, 4), w // 2 + np.random.randint(-3, 4))
        radius = np.random.randint(4, max(6, min(spatial_size) // 4))

        z, y, x = np.ogrid[:d, :h, :w]
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        tumor_mask = (dist <= radius).astype(np.float32)

        # WT (Whole Tumor)
        label[1] = tumor_mask
        # TC (Tumor Core)
        label[0] = (dist <= radius * 0.7).astype(np.float32)
        # ET (Enhancing Tumor)
        label[2] = (dist <= radius * 0.4).astype(np.float32)

        # Modulate image intensity in tumor area
        image[:, tumor_mask > 0] += 1.5

        img_file = out_path / f"synthetic_mri_{i:03d}_image.npy"
        lbl_file = out_path / f"synthetic_mri_{i:03d}_label.npy"

        np.save(img_file, image)
        np.save(lbl_file, label)

        data_list.append({"image": str(img_file), "label": str(lbl_file)})

    return data_list


# Hospital DataLoader Factory
def create_hospital_dataloaders(
    hospital_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    data_list: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 2,
    train_val_split: float = 0.8,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create training and validation DataLoaders for a specific hospital node.

    Returns:
        (train_loader, val_loader, num_train_samples, num_val_samples)
    """
    if data_list is None:
        if data_dir is None:
            raise ValueError("Either data_dir or data_list must be provided.")
        data_list = scan_local_dataset(data_dir)

    if not data_list:
        raise ValueError(f"No MRI data found for hospital '{hospital_id}' in '{data_dir}'")

    # Split Train / Validation
    np.random.seed(seed)
    n_samples = len(data_list)
    indices = np.random.permutation(n_samples)
    n_train = max(1, int(n_samples * train_val_split))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:] if n_samples > 1 else indices[:1]

    train_items = [data_list[i] for i in train_indices]
    val_items = [data_list[i] for i in val_indices]

    train_transforms = get_train_transforms(roi_size=roi_size)
    val_transforms = get_val_transforms(roi_size=roi_size)

    train_dataset = FedMedMRIDataset(train_items, transform=train_transforms)
    val_dataset = FedMedMRIDataset(val_items, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, len(train_items), len(val_items)
