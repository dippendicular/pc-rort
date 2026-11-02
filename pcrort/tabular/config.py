"""Experiment configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

METHODS = ["LOCAL", "POOLED", "UNIFORM", "FedAvg", "DIAG", "FULL", "STRUCTURED", "ADAPTIVE"]
MIXTURES = ["DIAG", "FULL", "STRUCTURED", "ADAPTIVE"]


@dataclass
class SplitConfig:
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    min_train_class: int = 6
    min_eval_class: int = 2


@dataclass
class ClientConfig:
    count: int = 3
    regimes: list[str] = field(default_factory=lambda: ["iid", "non_iid"])
    dirichlet_alpha: float = 0.5
    min_client_class: int = 2


@dataclass
class FedAvgConfig:
    rounds: int = 100
    local_epochs: int = 3
    learning_rate: float = 0.1


@dataclass
class MixtureConfig:
    external_mass_cap: float = 0.5
    radius_fraction: float = 0.03
    adaptive_folds: int = 5
    adaptive_penalty: float = 1.0
    max_iterations: int = 30


@dataclass
class ThresholdConfig:
    fixed: float = 0.5
    target_specificity: float = 0.9
    min_pilot_per_class: int = 5


@dataclass
class AnalysisConfig:
    primary_methods: list[str] = field(default_factory=lambda: ["FULL", "ADAPTIVE"])
    comparators: list[str] = field(default_factory=lambda: ["LOCAL", "FedAvg", "POOLED"])
    alpha: float = 0.05


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    datasets: list | str = "auto"
    dataset_root: str = "datasets"
    repetitions: int = 100
    seed: int = 2026110301
    output_dir: str = "outputs"
    l2: float = 0.01
    workers: int = 1
    save_arrays: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    clients: ClientConfig = field(default_factory=ClientConfig)
    fedavg: FedAvgConfig = field(default_factory=FedAvgConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def protocol(self) -> dict:
        """Settings that determine results; a resumed run must match them exactly."""
        out = asdict(self)
        for key in ("workers", "output_dir", "dataset_root"):
            out.pop(key)
        out["split"]["ratios"] = list(out["split"]["ratios"])
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        raw = dict(raw)
        sections = {"split": SplitConfig, "clients": ClientConfig, "fedavg": FedAvgConfig,
                    "mixture": MixtureConfig, "threshold": ThresholdConfig,
                    "analysis": AnalysisConfig}
        for key, kind in sections.items():
            if key in raw:
                raw[key] = kind(**raw[key])
        if isinstance(raw.get("split"), SplitConfig):
            raw["split"].ratios = tuple(raw["split"].ratios)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown experiment keys {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "ExperimentConfig":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(raw)
