"""Genomics dataset — loads gene expression vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GenomicsDataset(Dataset):
    """
    Dataset for gene expression data.

    Supports two input modes:

    1. **Per-patient files**: Each patient has an individual expression file
       (CSV/TSV) pointed to by a column in the metadata CSV.

    2. **Expression matrix**: A single matrix file (genes × patients or
       patients × genes) with a config flag for transposing.

    Expression data is preprocessed by GenomicsPreprocessor (fitted on
    training data). This dataset receives already-transformed numpy arrays.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        expression_data: np.ndarray,
        patient_id_col: str,
        label_col: str,
        num_classes: int = 2,
    ):
        """
        Args:
            metadata_df: DataFrame with patient metadata (aligned with expression_data rows).
            expression_data: Preprocessed expression matrix (n_samples, n_features).
            patient_id_col: Column for patient ID.
            label_col: Column for label.
            num_classes: Number of classes.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.expression_data = expression_data
        self.patient_id_col = patient_id_col
        self.label_col = label_col
        self.num_classes = num_classes

        if len(self.df) != len(self.expression_data):
            raise ValueError(
                f"Metadata ({len(self.df)} rows) and expression data "
                f"({len(self.expression_data)} rows) must have same length."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        patient_id = row[self.patient_id_col]
        label = int(row[self.label_col])

        # Expression vector
        expression = torch.tensor(
            self.expression_data[idx], dtype=torch.float32
        )

        # Label
        if self.num_classes == 2:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "patient_id": patient_id,
            "expression": expression,
            "label": label_tensor,
        }

    @property
    def input_dim(self) -> int:
        """Number of expression features."""
        return self.expression_data.shape[1]


def load_expression_matrix(
    metadata_df: pd.DataFrame,
    patient_id_col: str,
    expression_matrix_path: Optional[str] = None,
    expression_file_col: Optional[str] = None,
    transpose: bool = False,
) -> pd.DataFrame:
    """
    Load expression data into a DataFrame aligned with metadata.

    Args:
        metadata_df: Patient metadata DataFrame.
        patient_id_col: Column for patient IDs.
        expression_matrix_path: Path to a single matrix file.
        expression_file_col: Column in metadata with per-patient file paths.
        transpose: Whether to transpose the matrix.

    Returns:
        DataFrame with patients as rows, genes as columns.
    """
    if expression_matrix_path:
        path = Path(expression_matrix_path)
        if path.suffix == ".tsv":
            expr_df = pd.read_csv(path, sep="\t", index_col=0)
        else:
            expr_df = pd.read_csv(path, index_col=0)

        if transpose:
            expr_df = expr_df.T

        # Align with metadata patient IDs
        patient_ids = metadata_df[patient_id_col].astype(str).tolist()
        expr_df.index = expr_df.index.astype(str)

        # Find common patients
        common = [pid for pid in patient_ids if pid in expr_df.index]
        if len(common) == 0:
            raise ValueError(
                "No matching patient IDs between metadata and expression matrix. "
                f"Metadata IDs sample: {patient_ids[:5]}, "
                f"Expression IDs sample: {expr_df.index[:5].tolist()}"
            )

        if len(common) < len(patient_ids):
            print(
                f"  WARNING: {len(patient_ids) - len(common)} patients "
                f"not found in expression matrix."
            )

        # Reorder to match metadata
        expr_df = expr_df.loc[common]
        return expr_df

    elif expression_file_col:
        # Load per-patient files
        all_data = {}
        for _, row in metadata_df.iterrows():
            pid = str(row[patient_id_col])
            fpath = row.get(expression_file_col)
            if pd.isna(fpath) or str(fpath).strip() == "":
                continue
            fpath = Path(str(fpath))
            try:
                if fpath.suffix == ".tsv":
                    data = pd.read_csv(fpath, sep="\t", index_col=0).squeeze()
                else:
                    data = pd.read_csv(fpath, index_col=0).squeeze()
                all_data[pid] = data
            except Exception as e:
                print(f"  WARNING: Failed to load expression for {pid}: {e}")

        if not all_data:
            raise ValueError("No expression data loaded from per-patient files.")

        expr_df = pd.DataFrame(all_data).T
        expr_df.index.name = patient_id_col
        return expr_df

    else:
        raise ValueError(
            "Must provide either `expression_matrix_path` or `expression_file_col`."
        )
