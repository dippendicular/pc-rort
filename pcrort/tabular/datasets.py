"""Discovery and loading of binary tabular datasets.

Put any CSV, TSV, Excel or headerless ``.data`` table under ``datasets/``.
For each file the loader infers the header, the binary target column, the
positive label, identifier columns and target copies. Files that cannot
support a binary prediction task are reported and skipped.

Inference can be overridden per file with a sidecar YAML next to the data
(``<file stem>.yaml`` or ``dataset.yaml`` in the same folder) or with keyword
arguments to :func:`load_dataset`. Recognized keys: ``name``, ``title``,
``target``, ``positive``, ``exclude``, ``categorical``, ``header``,
``columns`` and ``na_values``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TABLE_SUFFIXES = {".csv": 0, ".data": 1, ".tsv": 2, ".xlsx": 3, ".xls": 4}
NA_VALUES = ["?", "NA", "N/A", "", "na", "null", "NULL", "None"]
MIN_ROWS = 30
MIN_MINORITY = 10

TARGET_NAME = re.compile(
    r"^(target|label|labels|y|class|classification|outcome|diagnosis|status|result|"
    r"severity|recurred|recurrence|response|event|death|survived|malignant|cancer|disease)$", re.I)
TARGET_HINT = re.compile(r"(target|label|class|outcome|diagnos)", re.I)
ID_NAME = re.compile(r"(^|[_\s])(id|identifier|patient[_\s]?id|record|index)$", re.I)
POSITIVE_TOKENS = {"1", "yes", "y", "true", "t", "positive", "pos", "m", "malignant", "cancer",
                   "case", "disease", "present", "abnormal", "recurred", "died", "dead", "event"}
NEGATIVE_TOKENS = {"0", "no", "n", "false", "f", "negative", "neg", "b", "benign", "control",
                   "healthy", "normal", "absent", "alive", "none"}
OVERRIDE_KEYS = {"name", "title", "target", "positive", "exclude", "categorical", "header",
                 "columns", "na_values"}


@dataclass
class DatasetSpec:
    """A data file plus any explicit choices that replace inference."""
    file: str
    name: str | None = None
    title: str | None = None
    target: str | None = None
    positive: list[str] | None = None
    exclude: dict[str, str] = field(default_factory=dict)
    categorical: list[str] = field(default_factory=list)
    header: bool | None = None
    columns: list[str] | None = None
    na_values: list[str] = field(default_factory=lambda: list(NA_VALUES))

    @classmethod
    def build(cls, file: str | Path, **options) -> "DatasetSpec":
        unknown = set(options) - OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"unknown dataset options {sorted(unknown)}")
        options = {k: v for k, v in options.items() if v is not None}
        if "positive" in options:
            value = options["positive"]
            options["positive"] = [str(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
        if isinstance(options.get("exclude"), (list, tuple)):
            options["exclude"] = {column: "excluded by user" for column in options["exclude"]}
        return cls(file=str(file), **options)


@dataclass
class TabularDataset:
    name: str
    x: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    metadata: dict

    @property
    def has_duplicates(self) -> bool:
        return len(np.unique(self.groups)) < len(self.groups)


class DatasetRejected(ValueError):
    """The file cannot be used as a binary prediction dataset."""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "dataset"


def sidecar_options(path: Path) -> dict:
    for candidate in (path.with_suffix(".yaml"), path.with_suffix(".yml"), path.parent / "dataset.yaml"):
        if candidate.exists():
            with open(candidate, encoding="utf-8") as handle:
                options = yaml.safe_load(handle) or {}
            return {k: v for k, v in options.items() if k in OVERRIDE_KEYS}
    return {}


def _numeric_share(values: pd.Series) -> float:
    present = values.dropna()
    if present.empty:
        return 0.0
    return float(pd.to_numeric(present, errors="coerce").notna().mean())


def detect_header(raw: pd.DataFrame) -> bool:
    """True when the first row looks like column names rather than a record."""
    first, rest = raw.iloc[0], raw.iloc[1:]
    if first.isna().any() or pd.to_numeric(first, errors="coerce").notna().any():
        return False
    numeric_below = any(_numeric_share(rest[c]) > 0.9 for c in raw)
    unseen_below = all(first[c] not in set(rest[c].dropna()) for c in raw)
    return numeric_below or unseen_below


def read_table(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        reader = lambda **kw: pd.read_excel(path, na_values=spec.na_values, **kw)
    else:
        separator = "\t" if suffix == ".tsv" else ","
        reader = lambda **kw: pd.read_csv(path, sep=separator, na_values=spec.na_values,
                                          skipinitialspace=True, **kw)
    header = spec.header
    if header is None and spec.columns is None:
        header = detect_header(reader(header=None, dtype=str, nrows=200))
    if header or (header is None and spec.columns is None):
        frame = reader(header=0)
        if spec.columns:
            frame.columns = spec.columns
    else:
        frame = reader(header=None)
        frame.columns = spec.columns or [f"column_{i + 1}" for i in range(frame.shape[1])]
    frame.columns = frame.columns.astype(str).str.strip()
    return frame.dropna(axis=0, how="all")


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def binary_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if frame[c].notna().all() and frame[c].nunique() == 2]


def infer_target(frame: pd.DataFrame) -> str:
    candidates = binary_columns(frame)
    if not candidates:
        raise DatasetRejected("no column with exactly two values and no missing entries")
    for pattern in (TARGET_NAME, TARGET_HINT):
        named = [c for c in candidates if pattern.search(c.strip())]
        if named:
            return named[-1]
    return candidates[-1]


def infer_positive(labels: pd.Series) -> str:
    values = sorted(labels.unique())
    lowered = {v: v.lower() for v in values}
    numbers = pd.to_numeric(pd.Series(values), errors="coerce")
    if numbers.notna().all():
        return values[int(np.argmax(numbers.to_numpy()))]
    for v in values:
        if lowered[v] in POSITIVE_TOKENS and not any(lowered[o] in POSITIVE_TOKENS for o in values if o != v):
            return v
    for v in values:
        other = [o for o in values if o != v][0]
        if lowered[other] in NEGATIVE_TOKENS and lowered[v] not in NEGATIVE_TOKENS:
            return v
    counts = labels.value_counts()
    return str(counts.idxmin())


def _categorical_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return str(value)


def _automatic_exclusions(x: pd.DataFrame, y: np.ndarray, protected: set[str]) -> dict[str, str]:
    excluded = {}
    for column in x:
        if column in protected:
            continue
        values = x[column]
        present = values.dropna()
        if present.nunique() <= 1:
            excluded[column] = "constant"
            continue
        integer_valued = pd.api.types.is_integer_dtype(values) or (
            pd.api.types.is_float_dtype(values) and np.all(np.mod(present, 1) == 0))
        all_unique = present.nunique() == len(values) and len(values) >= 20
        if all_unique and (integer_valued or ID_NAME.search(column) or not pd.api.types.is_numeric_dtype(values)):
            excluded[column] = "identifier (unique per row)"
            continue
        if values.notna().all() and values.nunique() == 2:
            codes = pd.factorize(values, sort=True)[0]
            if np.array_equal(codes, y) or np.array_equal(codes, 1 - y):
                excluded[column] = "identical to the target"
    return excluded


def load_dataset(file: str | Path | DatasetSpec, **options) -> TabularDataset:
    """Load one table and infer anything the sidecar or ``options`` leave open."""
    if isinstance(file, DatasetSpec):
        spec = file
    else:
        path = Path(file)
        spec = DatasetSpec.build(path, **{**sidecar_options(path), **options})
    path = Path(spec.file)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = read_table(path, spec)
    except Exception as exc:
        raise DatasetRejected(f"unreadable table ({exc})") from exc
    if len(frame) < MIN_ROWS or frame.shape[1] < 2:
        raise DatasetRejected(f"too small ({len(frame)} rows, {frame.shape[1]} columns)")

    target = spec.target or infer_target(frame)
    if target not in frame:
        raise DatasetRejected(f"target column {target!r} not found")
    if frame[target].isna().any():
        frame = frame[frame[target].notna()]
    labels = _labels(frame[target])
    if labels.nunique() != 2:
        raise DatasetRejected(f"target {target!r} has {labels.nunique()} classes, expected 2")
    positive = spec.positive or [infer_positive(labels)]
    y = labels.isin(positive).to_numpy(dtype=int)
    minority = int(min(y.sum(), len(y) - y.sum()))
    if minority < MIN_MINORITY:
        raise DatasetRejected(f"minority class has only {minority} rows")

    x = frame.drop(columns=[target]).reset_index(drop=True)
    excluded = {target: "target"}
    for column, reason in spec.exclude.items():
        if column in x:
            excluded[column] = reason
            x = x.drop(columns=column)
    automatic = _automatic_exclusions(x, y, set(spec.categorical))
    excluded.update(automatic)
    x = x.drop(columns=list(automatic))
    for column in spec.categorical:
        if column in x:
            x[column] = x[column].map(_categorical_value)
    if x.shape[1] == 0:
        raise DatasetRejected("no feature columns remain after exclusions")

    # Identical feature rows form one group, so a split never puts copies of a
    # pattern on both sides.
    groups = pd.util.hash_pandas_object(x.astype(str), index=False).to_numpy()
    unique = len(np.unique(groups))
    name = spec.name or slug(path.parent.name if path.parent.name not in {"", "."} else path.stem)
    other_label = [v for v in sorted(labels.unique()) if v not in positive][0]
    metadata = dict(
        name=name, title=spec.title or name, file=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target=target, positive_label=positive, negative_label=other_label,
        target_inferred=spec.target is None, positive_inferred=spec.positive is None,
        rows=len(y), positives=int(y.sum()), negatives=int(len(y) - y.sum()),
        prevalence=float(y.mean()), feature_columns=list(x.columns), excluded_columns=excluded,
        categorical_columns=list(x.select_dtypes(exclude=np.number).columns),
        missing_by_column={c: int(v) for c, v in x.isna().sum().items() if v},
        duplicate_feature_rows=len(y) - unique)
    return TabularDataset(name, x, y, groups, metadata)


@dataclass
class DiscoveryResult:
    file: Path
    name: str | None
    usable: bool
    reason: str = ""
    metadata: dict = field(default_factory=dict)


def candidate_files(root: str | Path) -> list[Path]:
    """Table files under ``root``; when one folder holds the same table in several
    formats (e.g. ``data.csv`` and ``data.xlsx``) only the preferred format is kept."""
    best: dict[tuple[Path, str], Path] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        if any(part.startswith((".", "_")) for part in path.relative_to(root).parts):
            continue
        key = (path.parent, path.stem.lower())
        if key not in best or TABLE_SUFFIXES[path.suffix.lower()] < TABLE_SUFFIXES[best[key].suffix.lower()]:
            best[key] = path
    return sorted(best.values())


def discover(root: str | Path = "datasets") -> list[DiscoveryResult]:
    results, seen = [], {}
    for path in candidate_files(root):
        try:
            data = load_dataset(path)
        except DatasetRejected as exc:
            results.append(DiscoveryResult(path, None, False, str(exc)))
            continue
        name = data.name
        if name in seen:
            seen[name] += 1
            name = f"{name}_{slug(path.stem)}"
        else:
            seen[name] = 1
        data.metadata["name"] = name
        results.append(DiscoveryResult(path, name, True, metadata=data.metadata))
    return results


def resolve_datasets(entries, root: str | Path = "datasets") -> list[DatasetSpec]:
    """Turn an experiment's ``datasets`` entry into specs.

    ``"auto"`` (or an empty list) means every usable file under ``root``. A list
    may mix file paths, dataset names found under ``root`` and mappings with a
    ``file`` key plus override options.
    """
    def from_discovery(result: DiscoveryResult) -> DatasetSpec:
        return DatasetSpec.build(result.file, **{**sidecar_options(result.file), "name": result.name})

    automatic = entries in (None, "auto", [], ["auto"])
    discovered = None
    if automatic:
        discovered = {r.name: r for r in discover(root) if r.usable}
        if not discovered:
            raise FileNotFoundError(f"no usable datasets found under {root}")
        return [from_discovery(r) for r in discovered.values()]

    specs = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = dict(entry)
            path = Path(entry.pop("file"))
            specs.append(DatasetSpec.build(path, **{**sidecar_options(path), **entry}))
        elif Path(entry).is_file():
            specs.append(DatasetSpec.build(entry, **sidecar_options(Path(entry))))
        else:
            if discovered is None:
                discovered = {r.name: r for r in discover(root) if r.usable}
            if entry not in discovered:
                raise FileNotFoundError(f"{entry!r} is neither a file nor a usable dataset under {root}")
            specs.append(from_discovery(discovered[entry]))
    return specs
