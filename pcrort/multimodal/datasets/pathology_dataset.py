"""
Pathology dataset — supports patch-based Multiple Instance Learning.

Each patient is a "bag" of patches. The dataset preserves the per-patient
bag structure required by Attention MIL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PathologyDataset(Dataset):
    """
    Patch-based pathology dataset for MIL.

    Supports two modes of discovering patches per patient:

    1. **Directory mode**: Each patient has a subdirectory under `patches_dir`
       containing patch image files.

    2. **Manifest mode**: A column in the metadata CSV lists comma-separated
       patch file paths.

    Each __getitem__ returns a bag of patch tensors for one patient.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        patient_id_col: str,
        label_col: str,
        patches_dir: Optional[str] = None,
        patches_manifest_col: Optional[str] = None,
        pathology_path_col: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_patches: int = 100,
        num_classes: int = 2,
    ):
        """
        Args:
            metadata_df: DataFrame with patient metadata.
            patient_id_col: Column for patient ID.
            label_col: Column for label.
            patches_dir: Base directory with per-patient patch subdirs.
            patches_manifest_col: Column with comma-separated patch paths.
            pathology_path_col: Column with path to a single pathology
                                image or patch directory per patient.
            transform: Torchvision transform for patches.
            max_patches: Maximum patches per patient bag.
            num_classes: Number of classes.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.patient_id_col = patient_id_col
        self.label_col = label_col
        self.patches_dir = patches_dir
        self.patches_manifest_col = patches_manifest_col
        self.pathology_path_col = pathology_path_col
        self.transform = transform
        self.max_patches = max_patches
        self.num_classes = num_classes

        # Determine mode
        if patches_manifest_col and patches_manifest_col in self.df.columns:
            self.mode = "manifest"
        elif patches_dir:
            self.mode = "directory"
        elif pathology_path_col and pathology_path_col in self.df.columns:
            self.mode = "single_path"
        else:
            raise ValueError(
                "PathologyDataset requires either `patches_dir`, "
                "`patches_manifest_col`, or `pathology_path_col` to be set."
            )

        # Pre-build the mapping from patient → patch list
        self._build_patch_map()

    def _build_patch_map(self) -> None:
        """Pre-compute patch file lists for each patient."""
        self.patch_map: Dict[str, List[str]] = {}
        valid_indices = []
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        for idx, row in self.df.iterrows():
            pid = str(row[self.patient_id_col])
            patches = []

            if self.mode == "directory":
                patient_dir = Path(self.patches_dir) / pid
                if patient_dir.is_dir():
                    patches = sorted([
                        str(p) for p in patient_dir.iterdir()
                        if p.suffix.lower() in image_extensions
                    ])

            elif self.mode == "manifest":
                manifest = row[self.patches_manifest_col]
                if pd.notna(manifest) and str(manifest).strip():
                    patches = [p.strip() for p in str(manifest).split(",") if p.strip()]

            elif self.mode == "single_path":
                path = row[self.pathology_path_col]
                if pd.notna(path) and str(path).strip():
                    path = Path(str(path))
                    if path.is_dir():
                        patches = sorted([
                            str(p) for p in path.iterdir()
                            if p.suffix.lower() in image_extensions
                        ])
                    elif path.is_file():
                        patches = [str(path)]

            if patches:
                self.patch_map[pid] = patches[:self.max_patches]
                valid_indices.append(idx)

        n_dropped = len(self.df) - len(valid_indices)
        if n_dropped > 0:
            print(f"  PathologyDataset: Dropped {n_dropped} patients with no patches.")
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        patient_id = str(row[self.patient_id_col])
        label = int(row[self.label_col])
        patch_paths = self.patch_map[patient_id]

        # Load and transform patches
        patches = []
        for ppath in patch_paths:
            try:
                img = Image.open(ppath).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                patches.append(img)
            except Exception as e:
                print(f"  WARNING: Failed to load patch {ppath}: {e}")
                continue

        if len(patches) == 0:
            raise RuntimeError(
                f"No valid patches loaded for patient {patient_id}. "
                f"Expected {len(patch_paths)} patches."
            )

        # Stack into bag tensor: (num_patches, C, H, W)
        bag = torch.stack(patches, dim=0)

        # Label
        if self.num_classes == 2:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "patient_id": patient_id,
            "patches": bag,
            "num_patches": len(patches),
            "label": label_tensor,
        }


def pathology_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for variable-size patch bags.

    Pads all bags to the maximum bag size in the batch.

    Returns:
        Dict with:
        - patches: (B, max_patches, C, H, W)
        - mask: (B, max_patches) — True for valid patches
        - labels: (B,)
        - patient_ids: list of str
        - num_patches: (B,)
    """
    max_patches = max(item["num_patches"] for item in batch)
    c, h, w = batch[0]["patches"].shape[1:]

    padded_patches = []
    masks = []
    labels = []
    patient_ids = []
    num_patches = []

    for item in batch:
        n = item["num_patches"]
        bag = item["patches"]  # (n, C, H, W)

        # Pad to max_patches
        if n < max_patches:
            padding = torch.zeros(max_patches - n, c, h, w)
            bag = torch.cat([bag, padding], dim=0)

        padded_patches.append(bag)
        mask = torch.zeros(max_patches, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)
        labels.append(item["label"])
        patient_ids.append(item["patient_id"])
        num_patches.append(n)

    return {
        "patient_ids": patient_ids,
        "patches": torch.stack(padded_patches, dim=0),
        "mask": torch.stack(masks, dim=0),
        "label": torch.stack(labels, dim=0),
        "num_patches": torch.tensor(num_patches, dtype=torch.long),
    }
