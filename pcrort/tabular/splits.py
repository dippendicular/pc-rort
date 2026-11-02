"""Train/pilot/test splitting and simulated client partitions."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .datasets import TabularDataset


def _fold_counts(ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    fractions = [Fraction(r).limit_denominator(20) for r in ratios]
    if sum(fractions) != 1:
        raise ValueError(f"split ratios must sum to one, got {ratios}")
    folds = 1
    for f in fractions:
        folds = folds * f.denominator // np.gcd(folds, f.denominator)
    return tuple(int(f * folds) for f in fractions)


def split_dataset(data: TabularDataset, seed: int, ratios=(0.6, 0.2, 0.2),
                  min_train_class: int = 6, min_eval_class: int = 2,
                  max_attempts: int = 100):
    """Return disjoint (train, pilot, test) index arrays and a split record.

    Datasets with duplicate feature rows are split by StratifiedGroupKFold so
    duplicate groups never cross partitions; others use two stratified splits.
    A split is redrawn with a new seed until the class-count minimums hold.
    """
    index = np.arange(len(data.y))
    train_folds, pilot_folds, test_folds = _fold_counts(tuple(ratios))
    for attempt in range(max_attempts):
        split_seed = seed + 10000 * attempt
        if data.has_duplicates:
            splitter = StratifiedGroupKFold(train_folds + pilot_folds + test_folds,
                                            shuffle=True, random_state=split_seed)
            folds = [fold for _, fold in splitter.split(index, data.y, data.groups)]
            test = np.concatenate(folds[:test_folds])
            pilot = np.concatenate(folds[test_folds:test_folds + pilot_folds])
            train = np.concatenate(folds[test_folds + pilot_folds:])
        else:
            train, test = train_test_split(index, test_size=ratios[2], stratify=data.y,
                                           random_state=split_seed)
            pilot_share = ratios[1] / (ratios[0] + ratios[1])
            train, pilot = train_test_split(train, test_size=pilot_share, stratify=data.y[train],
                                            random_state=split_seed + 1000)
        counts = [np.bincount(data.y[part], minlength=2) for part in (train, pilot, test)]
        if counts[0].min() < min_train_class or min(counts[1].min(), counts[2].min()) < min_eval_class:
            continue
        for a, b in [(train, pilot), (train, test), (pilot, test)]:
            assert not set(a) & set(b)
            assert not set(data.groups[a]) & set(data.groups[b])
        parts = tuple(np.asarray(part, int) for part in (train, pilot, test))
        return parts, dict(seed=split_seed, attempt=attempt, group_aware=data.has_duplicates,
                           ratios=list(ratios), sizes=[len(p) for p in parts],
                           class_counts=[c.tolist() for c in counts])
    raise ValueError(f"{data.name}: no split satisfies the class-count minimums")


def assign_clients(y, regime: str, seed: int, clients: int = 3, alpha: float = 0.5,
                   min_client_class: int = 2, max_attempts: int = 10000):
    """Partition training rows into clients, separately within each class.

    ``iid`` splits each class equally; ``non_iid`` draws within-class client
    shares from Dirichlet(alpha). Allocations are redrawn until every client
    holds at least ``min_client_class`` rows of each class.
    """
    if regime not in {"iid", "non_iid"}:
        raise ValueError(f"unknown regime {regime!r}")
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    for attempt in range(max_attempts):
        client_id = np.empty(len(y), int)
        for label in (0, 1):
            rows = np.flatnonzero(y == label)
            rng.shuffle(rows)
            if regime == "iid":
                parts = np.array_split(rows, clients)
            else:
                sizes = rng.multinomial(len(rows), rng.dirichlet(np.full(clients, alpha)))
                parts = np.split(rows, np.cumsum(sizes)[:-1])
            for k, part in enumerate(parts):
                client_id[part] = k
        counts = np.array([np.bincount(y[client_id == k], minlength=2) for k in range(clients)])
        if counts.min() >= min_client_class:
            return client_id, dict(attempt=attempt, class_counts=counts.tolist(),
                                   sizes=counts.sum(1).tolist())
    raise ValueError("cannot draw a client partition satisfying the class minimum")
