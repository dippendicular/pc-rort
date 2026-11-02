"""Loss functions with class weight support."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def build_loss(
    num_classes: int = 2,
    class_weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build the appropriate loss function.

    Args:
        num_classes: Number of classes (2 = binary).
        class_weights: Optional list of per-class weights.
        device: Device for weight tensors.

    Returns:
        Loss module.
    """
    if num_classes == 2:
        # Binary classification
        if class_weights is not None and len(class_weights) == 2:
            # BCEWithLogitsLoss uses pos_weight for the positive class
            pos_weight = torch.tensor(
                [class_weights[1] / class_weights[0]], device=device
            )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()
    else:
        # Multiclass
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()
