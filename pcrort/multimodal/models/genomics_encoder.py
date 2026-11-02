"""
Genomics encoder — MLP for gene expression data.

Architecture:
    Expression vector → LayerNorm → MLP (input→1024→512→256) → Embedding
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class GenomicsEncoder(nn.Module):
    """
    MLP encoder for gene expression vectors.

    The input dimension is dynamically configurable (not fixed).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        activation: str = "gelu",
        normalization: str = "layernorm",
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            input_dim: Number of input gene features.
            hidden_dims: List of hidden layer dimensions.
            embedding_dim: Final embedding dimension.
            dropout: Dropout rate.
            activation: Activation function (gelu, relu, leaky_relu).
            normalization: Normalization type (layernorm, batchnorm).
            num_classes: If set, adds classifier head.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Activation function
        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        }.get(activation, nn.GELU())

        # Build MLP
        layers = []

        # Input normalization
        if normalization == "layernorm":
            layers.append(nn.LayerNorm(input_dim))
        elif normalization == "batchnorm":
            layers.append(nn.BatchNorm1d(input_dim))

        # Hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection to embedding_dim (if last hidden != embedding_dim)
        if prev_dim != embedding_dim:
            layers.append(nn.Linear(prev_dim, embedding_dim))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(embedding_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(embedding_dim))
            layers.append(act_fn)

        self.mlp = nn.Sequential(*layers)

        # Optional classifier
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(
                embedding_dim, 1 if num_classes == 2 else num_classes
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from gene expression vector.

        Args:
            x: Expression tensor (B, input_dim).

        Returns:
            Embedding tensor (B, embedding_dim).
        """
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional classification.

        Args:
            x: Expression tensor (B, input_dim).

        Returns:
            Dict with 'embedding' and optionally 'logits'.
        """
        embedding = self.forward_features(x)
        output = {"embedding": embedding}

        if self.classifier is not None:
            logits = self.classifier(embedding)
            output["logits"] = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

        return output
