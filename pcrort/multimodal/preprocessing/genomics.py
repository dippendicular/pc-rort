"""
Genomics preprocessing pipeline.

Supports:
- Missing value imputation
- Log1p transformation
- Z-score standardization
- Variance-based feature selection

IMPORTANT: Fitted parameters are computed on training data only
and must be saved/reused for validation and test sets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class GenomicsPreprocessor:
    """
    Genomics expression data preprocessor.

    Fits on training data and transforms any split consistently.
    Fitted state can be saved/loaded for reproducibility.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor from genomics config.

        Args:
            config: Genomics preprocessing config section.
        """
        preproc = config.get("preprocessing", {})
        self.handle_missing = preproc.get("handle_missing", "median")
        self.log_transform = preproc.get("log_transform", True)
        self.standardize = preproc.get("standardize", True)

        fs_config = preproc.get("feature_selection", {})
        self.feature_selection_enabled = fs_config.get("enabled", False)
        self.fs_method = fs_config.get("method", "variance")
        self.min_variance = fs_config.get("min_variance", 0.1)
        self.top_k = fs_config.get("top_k", None)

        # Fitted parameters (set during fit)
        self.is_fitted = False
        self.impute_values_: Optional[pd.Series] = None
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, expression_df: pd.DataFrame) -> "GenomicsPreprocessor":
        """
        Fit the preprocessor on training expression data.

        Args:
            expression_df: DataFrame of gene expression values.
                           Rows = samples, Columns = genes.

        Returns:
            self (for chaining).
        """
        df = expression_df.copy()
        self.feature_names_ = list(df.columns)

        # Step 1: Compute imputation values
        if self.handle_missing == "median":
            self.impute_values_ = df.median()
        elif self.handle_missing == "mean":
            self.impute_values_ = df.mean()
        elif self.handle_missing == "zero":
            self.impute_values_ = pd.Series(0.0, index=df.columns)
        elif self.handle_missing == "drop":
            self.impute_values_ = None
        else:
            raise ValueError(f"Unknown missing handler: {self.handle_missing}")

        # Apply imputation for subsequent steps
        if self.impute_values_ is not None:
            df = df.fillna(self.impute_values_)

        # Validate numeric
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Non-numeric columns found in expression data: {non_numeric}"
            )

        # Step 2: Log transform (compute on already-imputed data)
        if self.log_transform:
            df = np.log1p(df.clip(lower=0))

        # Step 3: Feature selection
        if self.feature_selection_enabled:
            if self.fs_method == "variance":
                variances = df.var()
                if self.top_k is not None:
                    top_features = variances.nlargest(self.top_k).index.tolist()
                    self.selected_features_ = top_features
                else:
                    self.selected_features_ = variances[
                        variances >= self.min_variance
                    ].index.tolist()
            elif self.fs_method == "mad":
                mads = df.apply(lambda x: np.median(np.abs(x - np.median(x))))
                if self.top_k is not None:
                    top_features = mads.nlargest(self.top_k).index.tolist()
                    self.selected_features_ = top_features
                else:
                    self.selected_features_ = mads[
                        mads >= self.min_variance
                    ].index.tolist()
            else:
                raise ValueError(f"Unknown feature selection method: {self.fs_method}")

            if len(self.selected_features_) == 0:
                raise ValueError(
                    "Feature selection removed all features! "
                    "Lower `min_variance` or increase `top_k`."
                )

            df = df[self.selected_features_]
            print(
                f"  Feature selection: {len(self.feature_names_)} → "
                f"{len(self.selected_features_)} features"
            )

        # Step 4: Compute standardization parameters
        if self.standardize:
            self.mean_ = df.mean()
            self.std_ = df.std().replace(0, 1)  # Avoid division by zero

        self.is_fitted = True
        return self

    def transform(self, expression_df: pd.DataFrame) -> np.ndarray:
        """
        Transform expression data using fitted parameters.

        Args:
            expression_df: DataFrame of gene expression values.

        Returns:
            Transformed numpy array.

        Raises:
            RuntimeError: If the preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "GenomicsPreprocessor must be fitted before transform. "
                "Call .fit() on training data first."
            )

        df = expression_df.copy()

        # Imputation
        if self.impute_values_ is not None:
            # Only impute columns present in fitted data
            common = df.columns.intersection(self.impute_values_.index)
            df[common] = df[common].fillna(self.impute_values_[common])

        # Log transform
        if self.log_transform:
            df = np.log1p(df.clip(lower=0))

        # Feature selection
        if self.feature_selection_enabled and self.selected_features_ is not None:
            # Handle missing features gracefully
            missing = set(self.selected_features_) - set(df.columns)
            if missing:
                for feat in missing:
                    df[feat] = 0.0
            df = df[self.selected_features_]

        # Standardization
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            common = df.columns.intersection(self.mean_.index)
            df[common] = (df[common] - self.mean_[common]) / self.std_[common]

        return df.values.astype(np.float32)

    def fit_transform(self, expression_df: pd.DataFrame) -> np.ndarray:
        """Fit on data and transform it."""
        self.fit(expression_df)
        return self.transform(expression_df)

    def get_input_dim(self) -> int:
        """Return the number of features after preprocessing."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        if self.selected_features_ is not None:
            return len(self.selected_features_)
        if self.feature_names_ is not None:
            return len(self.feature_names_)
        raise RuntimeError("Cannot determine input dimension.")

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted preprocessor state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "handle_missing": self.handle_missing,
            "log_transform": self.log_transform,
            "standardize": self.standardize,
            "feature_selection_enabled": self.feature_selection_enabled,
            "fs_method": self.fs_method,
            "min_variance": self.min_variance,
            "top_k": self.top_k,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names_,
            "selected_features": self.selected_features_,
        }
        if self.impute_values_ is not None:
            state["impute_values"] = self.impute_values_.to_dict()
        if self.mean_ is not None:
            state["mean"] = self.mean_.to_dict()
        if self.std_ is not None:
            state["std"] = self.std_.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenomicsPreprocessor":
        """Load a fitted preprocessor from disk."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Reconstruct config-like dict
        config = {
            "preprocessing": {
                "handle_missing": state["handle_missing"],
                "log_transform": state["log_transform"],
                "standardize": state["standardize"],
                "feature_selection": {
                    "enabled": state["feature_selection_enabled"],
                    "method": state["fs_method"],
                    "min_variance": state["min_variance"],
                    "top_k": state["top_k"],
                },
            }
        }
        preprocessor = cls(config)
        preprocessor.is_fitted = state["is_fitted"]
        preprocessor.feature_names_ = state.get("feature_names")
        preprocessor.selected_features_ = state.get("selected_features")

        if "impute_values" in state:
            preprocessor.impute_values_ = pd.Series(state["impute_values"])
        if "mean" in state:
            preprocessor.mean_ = pd.Series(state["mean"])
        if "std" in state:
            preprocessor.std_ = pd.Series(state["std"])

        return preprocessor
