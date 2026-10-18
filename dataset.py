"""
dataset.py — APTOS 2019 Dataset Loader
========================================
PyTorch Dataset for the APTOS 2019 Blindness Detection dataset.

Handles:
  - CSV parsing and image loading
  - Stratified train/val/test splitting
  - On-the-fly preprocessing and augmentation
  - Class weight computation for imbalanced data

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import PATHS, DATA
from src.preprocessing import preprocess_fundus, get_train_transforms, get_val_transforms


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy fundus images.

    Loads images from disk, applies preprocessing pipeline,
    and returns (image_tensor, label) pairs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform=None,
        apply_preprocessing: bool = True,
    ):
        """
        Args:
            df: DataFrame with columns ['id_code', 'diagnosis'].
            image_dir: Directory containing the fundus images.
            transform: Albumentations transform pipeline.
            apply_preprocessing: Whether to apply fundus preprocessing.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.apply_preprocessing = apply_preprocessing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row["id_code"]
        label = int(row["diagnosis"])

        # Load image
        from src.utils import safe_imread
        image_path = self.image_dir / f"{image_id}.png"
        image = safe_imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Apply fundus-specific preprocessing
        if self.apply_preprocessing:
            image = preprocess_fundus(image)

        # Convert BGR → RGB for albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation/normalization transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def create_splits(
    csv_path: Path,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Args:
        csv_path: Path to APTOS train.csv.
        val_ratio: Validation set ratio (default from config).
        test_ratio: Test set ratio (default from config).
        seed: Random seed (default from config).

    Returns:
        (train_df, val_df, test_df) — three DataFrames.
    """
    val_ratio = val_ratio or DATA.val_ratio
    test_ratio = test_ratio or DATA.test_ratio
    seed = seed or DATA.random_seed

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed,
        stratify=df["diagnosis"]
    )

    # Second split: train vs val
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_ratio, random_state=seed,
        stratify=train_val_df["diagnosis"]
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced data.

    Args:
        df: DataFrame with 'diagnosis' column.

    Returns:
        Tensor of shape (num_classes,) with class weights.
    """
    counts = df["diagnosis"].value_counts().sort_index().values.astype(float)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights)


def get_dataloaders(
    csv_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test DataLoaders.

    Args:
        csv_path: Path to train.csv (default: PATHS.data_raw / DATA.csv_file).
        image_dir: Path to image directory (default: PATHS.data_raw / DATA.image_dir).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    csv_path = csv_path or PATHS.data_raw / DATA.csv_file
    image_dir = image_dir or PATHS.data_raw / DATA.image_dir

    # Create splits
    train_df, val_df, test_df = create_splits(csv_path)

    # Save splits for reproducibility
    PATHS.data_splits.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PATHS.data_splits / "train.csv", index=False)
    val_df.to_csv(PATHS.data_splits / "val.csv", index=False)
    test_df.to_csv(PATHS.data_splits / "test.csv", index=False)

    # Create datasets
    train_dataset = DRDataset(train_df, image_dir, transform=get_train_transforms())
    val_dataset = DRDataset(val_df, image_dir, transform=get_val_transforms())
    test_dataset = DRDataset(test_df, image_dir, transform=get_val_transforms())

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA.batch_size,
        shuffle=True,
        num_workers=DATA.num_workers,
        pin_memory=DATA.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DATA.batch_size,
        shuffle=False,
        num_workers=DATA.num_workers,
        pin_memory=DATA.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=DATA.batch_size,
        shuffle=False,
        num_workers=DATA.num_workers,
        pin_memory=DATA.pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Dataset module loaded successfully.")
    print(f"Expected CSV: {PATHS.data_raw / DATA.csv_file}")
    print(f"Expected images: {PATHS.data_raw / DATA.image_dir}")
    print(f"Num classes: {DATA.num_classes}")
    print(f"Class names: {DATA.class_names}")
