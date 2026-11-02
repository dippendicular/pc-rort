"""
Multimodal dataset — loads radiology, pathology, and genomics per patient.

Handles patient-level alignment and optional missing modalities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    Patient-level multimodal dataset.

    Loads radiology images, pathology patch bags, and genomics expression
    vectors for each patient. All column names are configurable.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        patient_id_col: str,
        label_col: str,
        num_classes: int = 2,
        # Radiology
        radiology_path_col: Optional[str] = None,
        radiology_transform: Optional[Callable] = None,
        # Pathology
        pathology_path_col: Optional[str] = None,
        patches_dir: Optional[str] = None,
        patches_manifest_col: Optional[str] = None,
        pathology_transform: Optional[Callable] = None,
        max_patches: int = 100,
        # Genomics
        genomics_expression: Optional[np.ndarray] = None,
        genomics_patient_ids: Optional[List[str]] = None,
        # Modality flags
        use_radiology: bool = True,
        use_pathology: bool = True,
        use_genomics: bool = True,
        modality_mode: str = "STRICT_MULTIMODAL",
    ):
        """
        Args:
            metadata_df: Patient metadata DataFrame.
            patient_id_col: Column for patient ID.
            label_col: Column for label.
            num_classes: Number of classes.
            radiology_path_col: Column for radiology image path.
            radiology_transform: Transform for radiology images.
            pathology_path_col: Column for pathology data path.
            patches_dir: Base directory for pathology patches.
            patches_manifest_col: Column for patch file manifest.
            pathology_transform: Transform for pathology patches.
            max_patches: Max patches per patient.
            genomics_expression: Preprocessed expression array (aligned).
            genomics_patient_ids: Patient IDs corresponding to expression rows.
            use_radiology: Whether to load radiology.
            use_pathology: Whether to load pathology.
            use_genomics: Whether to load genomics.
            modality_mode: STRICT_MULTIMODAL or ALLOW_MISSING_MODALITIES.
        """
        self.patient_id_col = patient_id_col
        self.label_col = label_col
        self.num_classes = num_classes
        self.radiology_path_col = radiology_path_col
        self.radiology_transform = radiology_transform
        self.pathology_path_col = pathology_path_col
        self.patches_dir = patches_dir
        self.patches_manifest_col = patches_manifest_col
        self.pathology_transform = pathology_transform
        self.max_patches = max_patches
        self.use_radiology = use_radiology
        self.use_pathology = use_pathology
        self.use_genomics = use_genomics
        self.modality_mode = modality_mode

        # Build genomics lookup
        self.genomics_lookup: Dict[str, np.ndarray] = {}
        if use_genomics and genomics_expression is not None and genomics_patient_ids is not None:
            for pid, expr in zip(genomics_patient_ids, genomics_expression):
                self.genomics_lookup[str(pid)] = expr

        # Build pathology patch map
        self.patch_map: Dict[str, List[str]] = {}
        if use_pathology:
            self._build_patch_map(metadata_df)

        # Filter based on modality mode
        self.df = self._filter_patients(metadata_df)

    def _build_patch_map(self, df: pd.DataFrame) -> None:
        """Build patient → patch paths mapping."""
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        for _, row in df.iterrows():
            pid = str(row[self.patient_id_col])
            patches = []

            if self.patches_manifest_col and self.patches_manifest_col in df.columns:
                manifest = row.get(self.patches_manifest_col)
                if pd.notna(manifest) and str(manifest).strip():
                    patches = [p.strip() for p in str(manifest).split(",")]

            elif self.patches_dir:
                patient_dir = Path(self.patches_dir) / pid
                if patient_dir.is_dir():
                    patches = sorted([
                        str(p) for p in patient_dir.iterdir()
                        if p.suffix.lower() in image_extensions
                    ])

            elif self.pathology_path_col and self.pathology_path_col in df.columns:
                path = row.get(self.pathology_path_col)
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

    def _filter_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter patients based on modality availability."""
        valid_indices = []
        for idx, row in df.iterrows():
            pid = str(row[self.patient_id_col])
            has_all = True

            if self.use_radiology:
                if self.radiology_path_col and self.radiology_path_col in df.columns:
                    val = row.get(self.radiology_path_col)
                    if pd.isna(val) or str(val).strip() == "":
                        has_all = False

            if self.use_pathology:
                if pid not in self.patch_map:
                    has_all = False

            if self.use_genomics:
                if pid not in self.genomics_lookup:
                    has_all = False

            if self.modality_mode == "STRICT_MULTIMODAL":
                if has_all:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        filtered = df.loc[valid_indices].reset_index(drop=True)
        n_dropped = len(df) - len(filtered)
        if n_dropped > 0:
            print(
                f"  MultimodalDataset: Dropped {n_dropped} patients "
                f"with missing modalities ({self.modality_mode})."
            )
        return filtered

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        pid = str(row[self.patient_id_col])
        label = int(row[self.label_col])

        sample: Dict[str, Any] = {
            "patient_id": pid,
        }

        # --- Radiology ---
        if self.use_radiology:
            img_path = str(row.get(self.radiology_path_col, ""))
            try:
                image = Image.open(img_path).convert("RGB")
                if self.radiology_transform:
                    image = self.radiology_transform(image)
                sample["radiology_image"] = image
            except Exception:
                sample["radiology_image"] = None

        # --- Pathology ---
        if self.use_pathology:
            patch_paths = self.patch_map.get(pid, [])
            patches = []
            for pp in patch_paths:
                try:
                    img = Image.open(pp).convert("RGB")
                    if self.pathology_transform:
                        img = self.pathology_transform(img)
                    patches.append(img)
                except Exception:
                    continue
            if patches:
                sample["pathology_patches"] = torch.stack(patches, dim=0)
                sample["num_patches"] = len(patches)
            else:
                sample["pathology_patches"] = None
                sample["num_patches"] = 0

        # --- Genomics ---
        if self.use_genomics:
            expr = self.genomics_lookup.get(pid)
            if expr is not None:
                sample["genomics_expression"] = torch.tensor(
                    expr, dtype=torch.float32
                )
            else:
                sample["genomics_expression"] = None

        # --- Label ---
        if self.num_classes == 2:
            sample["label"] = torch.tensor(label, dtype=torch.float32)
        else:
            sample["label"] = torch.tensor(label, dtype=torch.long)

        return sample


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate for multimodal batches with variable-size pathology bags.

    Returns a dict with batched tensors and metadata.
    """
    collated: Dict[str, Any] = {
        "patient_ids": [item["patient_id"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
    }

    # A batch must never silently lose samples: use a zero placeholder plus an
    # availability mask.  PC-RORT records the absence as an evidence gap later.
    first_rad = next((x.get("radiology_image") for x in batch if x.get("radiology_image") is not None), None)
    if first_rad is not None:
        collated["radiology_image"] = torch.stack([x.get("radiology_image") if x.get("radiology_image") is not None else torch.zeros_like(first_rad) for x in batch])
        collated["radiology_available"] = torch.tensor([x.get("radiology_image") is not None for x in batch], dtype=torch.bool)

    # Pathology (pad bags)
    if any("pathology_patches" in item for item in batch):
        valid_bags = [(item["pathology_patches"], item["num_patches"]) for item in batch if item.get("pathology_patches") is not None]
        if valid_bags:
            max_p = max(n for _, n in valid_bags)
            c, h, w = valid_bags[0][0].shape[1:]
            padded = []
            masks = []
            num_patches_list = []
            for item in batch:
                bag, n = item.get("pathology_patches"), item.get("num_patches", 0)
                if bag is None:
                    bag = torch.zeros(0, c, h, w)
                    n = 0
                if n < max_p:
                    pad = torch.zeros(max_p - n, c, h, w)
                    bag = torch.cat([bag, pad], dim=0)
                padded.append(bag)
                mask = torch.zeros(max_p, dtype=torch.bool)
                mask[:n] = True
                masks.append(mask)
                num_patches_list.append(n)
            collated["pathology_patches"] = torch.stack(padded, dim=0)
            collated["pathology_mask"] = torch.stack(masks, dim=0)
            collated["num_patches"] = torch.tensor(num_patches_list)
            collated["pathology_available"] = collated["pathology_mask"].any(dim=1)

    # Genomics
    first_expr = next((x.get("genomics_expression") for x in batch if x.get("genomics_expression") is not None), None)
    if first_expr is not None:
        collated["genomics_expression"] = torch.stack([x.get("genomics_expression") if x.get("genomics_expression") is not None else torch.zeros_like(first_expr) for x in batch])
        collated["genomics_available"] = torch.tensor([x.get("genomics_expression") is not None for x in batch], dtype=torch.bool)

    return collated
