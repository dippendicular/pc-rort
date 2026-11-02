"""Federated preprocessing from per-client aggregate statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


class FederatedPreprocessor:
    """Impute, standardize and one-hot encode using only client summaries.

    Each client reports count, sum and sum of squares for numeric columns and
    category counts for categorical columns. The server combines them; no
    row-level data leaves a client.
    """

    def fit(self, x: pd.DataFrame, client_id: np.ndarray) -> "FederatedPreprocessor":
        self.numeric = list(x.select_dtypes(include=np.number).columns)
        self.categorical = [c for c in x if c not in self.numeric]
        self.summaries = []
        for k in np.unique(client_id):
            local = x.iloc[np.flatnonzero(client_id == k)]
            entry = {"client": int(k), "n": len(local), "numeric": {}, "categorical": {}}
            for col in self.numeric:
                v = local[col].dropna().to_numpy(float)
                entry["numeric"][col] = {"count": len(v), "sum": float(v.sum()),
                                         "sum_square": float(v @ v)}
            for col in self.categorical:
                entry["categorical"][col] = local[col].dropna().astype(str).value_counts().to_dict()
            self.summaries.append(entry)

        n = len(x)
        self.mean, self.scale = {}, {}
        for col in self.numeric:
            c = sum(e["numeric"][col]["count"] for e in self.summaries)
            s = sum(e["numeric"][col]["sum"] for e in self.summaries)
            ss = sum(e["numeric"][col]["sum_square"] for e in self.summaries)
            mean = s / c if c else 0.0
            # Variance after mean imputation of the n - c missing entries.
            variance = max(0.0, (ss + (n - c) * mean ** 2) / n - mean ** 2)
            self.mean[col] = mean
            self.scale[col] = np.sqrt(variance) if variance > 1e-20 else 1.0

        self.modes, self.levels = {}, {}
        for col in self.categorical:
            totals = {}
            for entry in self.summaries:
                for value, count in entry["categorical"][col].items():
                    totals[value] = totals.get(value, 0) + count
            mode = sorted(totals, key=lambda v: (-totals[v], v))[0] if totals else "__MISSING__"
            self.modes[col] = mode
            self.levels[col] = sorted(totals) if totals else [mode]
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        columns = []
        for col in self.numeric:
            values = x[col].fillna(self.mean[col]).to_numpy(float)
            columns.append(((values - self.mean[col]) / self.scale[col])[:, None])
        for col in self.categorical:
            values = x[col].fillna(self.modes[col]).astype(str).to_numpy()
            columns.append(np.column_stack([values == v for v in self.levels[col]]).astype(float))
        matrix = np.column_stack([*columns, np.ones((len(x), 1))])
        if not np.isfinite(matrix).all():
            raise ValueError("preprocessed features contain non-finite values")
        return matrix

    def config(self) -> dict:
        return dict(numeric=self.numeric, categorical=self.categorical, mean=self.mean,
                    scale=self.scale, modes=self.modes, levels=self.levels,
                    client_summaries=self.summaries)
