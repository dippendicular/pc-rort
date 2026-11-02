"""
Federated Learning aggregation strategies (FedAvg, FedProx, etc).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def aggregate_fedavg(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Federated Averaging (FedAvg).

    Args:
        results: List of tuples (parameters, num_samples).

    Returns:
        Aggregated parameters.
    """
    total_samples = sum(num_samples for _, num_samples in results)
    
    # Initialize aggregated weights with zeros
    aggregated_weights = [np.zeros_like(w) for w in results[0][0]]
    
    for parameters, num_samples in results:
        weight = num_samples / total_samples
        for i, param in enumerate(parameters):
            aggregated_weights[i] += param * weight
            
    return aggregated_weights


def aggregate_state_dicts(results, shared_keys):
    """FedAvg only named shared parameters; site-private state never leaves clients."""
    total = sum(n for _, n in results)
    if not total:
        raise ValueError("Cannot aggregate an empty client cohort")
    return {key: sum(state[key].float() * (n / total) for state, n in results) for key in shared_keys}
