"""
Patient-level metadata validation, matching, and splitting.

All column names are read from config — nothing is hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_metadata(
    csv_path: Union[str, Path],
    patient_id_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Load and validate a patient-level metadata CSV.

    Args:
        csv_path: Path to the metadata CSV.
        patient_id_col: Column name for patient identifiers.
        label_col: Column name for classification labels.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    missing_cols = []
    for col in [patient_id_col, label_col]:
        if col not in df.columns:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in metadata CSV: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure patient IDs are unique
    if df[patient_id_col].duplicated().any():
        n_dup = df[patient_id_col].duplicated().sum()
        print(f"  WARNING: {n_dup} duplicate patient IDs found. Keeping first occurrence.")
        df = df.drop_duplicates(subset=[patient_id_col], keep="first")

    # Drop rows with missing labels
    n_missing_label = df[label_col].isna().sum()
    if n_missing_label > 0:
        print(f"  WARNING: {n_missing_label} rows with missing labels removed.")
        df = df.dropna(subset=[label_col])

    return df.reset_index(drop=True)


def print_dataset_statistics(
    df: pd.DataFrame,
    patient_id_col: str,
    label_col: str,
    radiology_path_col: Optional[str] = None,
    pathology_path_col: Optional[str] = None,
    genomics_path_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Print and return dataset statistics.

    Args:
        df: Patient-level metadata DataFrame.
        patient_id_col: Column name for patient IDs.
        label_col: Column name for labels.
        radiology_path_col: Optional column for radiology paths.
        pathology_path_col: Optional column for pathology paths.
        genomics_path_col: Optional column for genomics paths.

    Returns:
        Dict of statistics.
    """
    stats: Dict[str, Any] = {}

    total = len(df)
    stats["total_patients"] = total

    # Class distribution
    class_dist = df[label_col].value_counts().to_dict()
    stats["class_distribution"] = class_dist

    print("\n" + "=" * 60)
    print(" DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total patients: {total}")
    print(f"  Class distribution:")
    for cls, count in sorted(class_dist.items(), key=lambda x: str(x[0])):
        pct = count / total * 100
        print(f"    {cls}: {count} ({pct:.1f}%)")

    # Modality availability
    modality_cols = {
        "radiology": radiology_path_col,
        "pathology": pathology_path_col,
        "genomics": genomics_path_col,
    }

    available_counts = {}
    missing_counts = {}
    for name, col in modality_cols.items():
        if col and col in df.columns:
            available = df[col].notna() & (df[col] != "")
            available_counts[name] = int(available.sum())
            missing_counts[name] = int((~available).sum())
        else:
            available_counts[name] = 0
            missing_counts[name] = total

    stats["modality_available"] = available_counts
    stats["modality_missing"] = missing_counts

    print(f"\n  Modality availability:")
    for name in ["radiology", "pathology", "genomics"]:
        avail = available_counts[name]
        miss = missing_counts[name]
        print(f"    {name.capitalize()}: {avail} available, {miss} missing")

    # All three modalities available
    all_available_mask = pd.Series([True] * total)
    for name, col in modality_cols.items():
        if col and col in df.columns:
            all_available_mask &= df[col].notna() & (df[col] != "")
        else:
            all_available_mask = pd.Series([False] * total)
            break

    all_three = int(all_available_mask.sum())
    stats["all_modalities_available"] = all_three
    print(f"\n  All three modalities available: {all_three}")
    print("=" * 60 + "\n")

    return stats


def validate_modalities(
    df: pd.DataFrame,
    modality_mode: str,
    radiology_path_col: Optional[str] = None,
    pathology_path_col: Optional[str] = None,
    genomics_path_col: Optional[str] = None,
    required_modalities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter dataset based on modality availability mode.

    Args:
        df: Patient-level metadata DataFrame.
        modality_mode: 'STRICT_MULTIMODAL' or 'ALLOW_MISSING_MODALITIES'.
        radiology_path_col: Column for radiology paths.
        pathology_path_col: Column for pathology paths.
        genomics_path_col: Column for genomics paths.
        required_modalities: List of required modalities (default: all).

    Returns:
        Filtered DataFrame.
    """
    if required_modalities is None:
        required_modalities = ["radiology", "pathology", "genomics"]

    col_map = {
        "radiology": radiology_path_col,
        "pathology": pathology_path_col,
        "genomics": genomics_path_col,
    }

    if modality_mode == "STRICT_MULTIMODAL":
        mask = pd.Series([True] * len(df), index=df.index)
        for modality in required_modalities:
            col = col_map.get(modality)
            if col and col in df.columns:
                mask &= df[col].notna() & (df[col].astype(str) != "")
            else:
                raise ValueError(
                    f"STRICT_MULTIMODAL requires '{modality}' but column "
                    f"'{col}' not found in metadata. "
                    f"Available columns: {list(df.columns)}"
                )
        filtered = df[mask].reset_index(drop=True)
        n_dropped = len(df) - len(filtered)
        if n_dropped > 0:
            print(
                f"  STRICT_MULTIMODAL: Dropped {n_dropped} patients "
                f"with missing required modalities."
            )
        return filtered

    elif modality_mode == "ALLOW_MISSING_MODALITIES":
        print("  ALLOW_MISSING_MODALITIES: Keeping all patients.")
        print("  NOTE: Missing modality handling is not yet fully implemented.")
        return df.copy()

    else:
        raise ValueError(
            f"Unknown modality_mode: '{modality_mode}'. "
            f"Use 'STRICT_MULTIMODAL' or 'ALLOW_MISSING_MODALITIES'."
        )


def create_patient_splits(
    df: pd.DataFrame,
    patient_id_col: str,
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
    splits_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create patient-level train/val/test splits.

    Ensures no patient appears in multiple splits.

    Args:
        df: Patient-level metadata DataFrame.
        patient_id_col: Column for patient IDs.
        label_col: Column for labels.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        stratify: Whether to use stratified splitting.
        seed: Random seed.
        splits_dir: Optional directory to save split CSVs.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    labels = df[label_col].values

    # First split: train+val vs test
    test_size = test_ratio
    if stratify:
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        train_val_idx, test_idx = next(splitter1.split(df, labels))
    else:
        n_total = len(df)
        indices = np.random.RandomState(seed).permutation(n_total)
        n_test = int(n_total * test_size)
        test_idx = indices[:n_test]
        train_val_idx = indices[n_test:]

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Second split: train vs val (within train+val)
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_val_labels = train_val_df[label_col].values

    if stratify:
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_relative, random_state=seed
        )
        train_idx, val_idx = next(splitter2.split(train_val_df, train_val_labels))
    else:
        n_tv = len(train_val_df)
        indices = np.random.RandomState(seed + 1).permutation(n_tv)
        n_val = int(n_tv * val_relative)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n  Split sizes:")
    print(f"    Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Verify no patient overlap
    train_ids = set(train_df[patient_id_col])
    val_ids = set(val_df[patient_id_col])
    test_ids = set(test_df[patient_id_col])

    assert len(train_ids & val_ids) == 0, "Patient overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Patient overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "Patient overlap between val and test!"

    # Save splits
    if splits_dir:
        splits_dir = Path(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(splits_dir / "val.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)
        print(f"  Splits saved to {splits_dir}")

    return train_df, val_df, test_df


def load_existing_splits(
    splits_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-existing train/val/test split CSVs.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    splits_dir = Path(splits_dir)
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    print(f"  Loaded existing splits from {splits_dir}")
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df
