"""Generic classification head."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    MLP classification head.

    Takes a fused embedding and produces class logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input embedding.
            num_classes: Number of output classes (2 = binary with single logit).
            hidden_dims: Optional list of hidden layer dims.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        if hidden_dims:
            for hdim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hdim

        # Output layer
        out_dim = 1 if num_classes == 2 else num_classes
        layers.append(nn.Linear(prev_dim, out_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embedding (B, input_dim).

        Returns:
            Logits: (B,) for binary, (B, num_classes) for multiclass.
        """
        logits = self.head(x)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)
        return logits
