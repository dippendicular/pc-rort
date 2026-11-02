"""Radiology dataset — loads medical images for single-modality training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RadiologyDataset(Dataset):
    """
    Dataset for radiology images.

    Each sample is a single image loaded from a path specified in the
    metadata CSV. Column names are fully configurable.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        patient_id_col: str,
        label_col: str,
        image_path_col: str,
        transform: Optional[Callable] = None,
        num_classes: int = 2,
    ):
        """
        Args:
            metadata_df: DataFrame with patient metadata.
            patient_id_col: Column name for patient ID.
            label_col: Column name for label.
            image_path_col: Column name for image file path.
            transform: Optional torchvision transform.
            num_classes: Number of classes (2 = binary).
        """
        self.df = metadata_df.reset_index(drop=True)
        self.patient_id_col = patient_id_col
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.transform = transform
        self.num_classes = num_classes

        # Validate that image path column exists
        if image_path_col not in self.df.columns:
            raise ValueError(
                f"Image path column '{image_path_col}' not found. "
                f"Available: {list(self.df.columns)}"
            )

        # Filter out rows with missing image paths
        valid_mask = self.df[image_path_col].notna() & (
            self.df[image_path_col].astype(str) != ""
        )
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            print(f"  RadiologyDataset: Dropped {n_dropped} rows with missing image paths.")
        self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        patient_id = row[self.patient_id_col]
        label = int(row[self.label_col])
        image_path = str(row[self.image_path_col])

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load radiology image for patient {patient_id}: "
                f"{image_path}. Error: {e}"
            )

        if self.transform:
            image = self.transform(image)

        # Label tensor
        if self.num_classes == 2:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "patient_id": patient_id,
            "image": image,
            "label": label_tensor,
        }
