"""
Fusion modules — Concatenation, Gated, and Attention fusion strategies.

All fusion modules:
- Accept projected modality embeddings
- Return fused representation and optional modality contribution weights
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion.

    Concatenates modality embeddings and projects to a common dimension.
    Modality weights are uniform (1/N for each modality).
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        projection_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            modality_dims: Dict mapping modality name → embedding dim.
            projection_dim: Output dimension after fusion.
            dropout: Dropout rate.
        """
        super().__init__()
        total_dim = sum(modality_dims.values())
        self.modality_names = sorted(modality_dims.keys())
        self.projection_dim = projection_dim

        self.projection = nn.Sequential(
            nn.Linear(total_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: Dict of modality name → embedding tensor (B, D_m).

        Returns:
            Tuple of:
            - fused: Fused representation (B, projection_dim).
            - weights: Uniform modality weights.
        """
        ordered = [embeddings[name] for name in self.modality_names]
        concatenated = torch.cat(ordered, dim=-1)
        fused = self.projection(concatenated)

        # Uniform weights
        n = len(self.modality_names)
        batch_size = fused.size(0)
        weights = {
            name: torch.full((batch_size,), 1.0 / n, device=fused.device)
            for name in self.modality_names
        }

        return fused, weights


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion.

    Each modality gets a learned gate (sigmoid) that controls its
    contribution to the fused representation.

    Returns modality-specific contribution weights for analysis.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        projection_dim: int = 256,
        dropout: float = 0.3,
        gate_activation: str = "sigmoid",
    ):
        """
        Args:
            modality_dims: Dict mapping modality name → embedding dim.
            projection_dim: Common latent space dimension.
            dropout: Dropout rate.
            gate_activation: 'sigmoid' or 'softmax'.
        """
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.projection_dim = projection_dim
        self.gate_activation = gate_activation

        # Per-modality gate networks
        self.gates = nn.ModuleDict()
        total_dim = sum(modality_dims.values())

        for name in self.modality_names:
            self.gates[name] = nn.Sequential(
                nn.Linear(total_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
            )

        # Per-modality projection to common space
        self.projections = nn.ModuleDict()
        for name, dim in sorted(modality_dims.items()):
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
                nn.GELU(),
            )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: Dict of modality name → embedding tensor (B, D_m).

        Returns:
            Tuple of:
            - fused: Fused representation (B, projection_dim).
            - weights: Modality gate values (B,) per modality.
        """
        # Concatenate all embeddings for gate input
        ordered = [embeddings[name] for name in self.modality_names]
        concat = torch.cat(ordered, dim=-1)  # (B, total_dim)

        # Compute gates
        gate_values = {}
        for name in self.modality_names:
            gate = self.gates[name](concat)  # (B, proj_dim)
            gate_values[name] = gate

        # Apply activation
        if self.gate_activation == "softmax":
            # Stack gates and apply softmax across modalities
            stacked = torch.stack(
                [gate_values[name] for name in self.modality_names], dim=0
            )  # (M, B, proj_dim)
            stacked = F.softmax(stacked, dim=0)
            for i, name in enumerate(self.modality_names):
                gate_values[name] = stacked[i]
        else:
            # Independent sigmoid gates
            for name in self.modality_names:
                gate_values[name] = torch.sigmoid(gate_values[name])

        # Project modalities and apply gates
        fused = torch.zeros(
            embeddings[self.modality_names[0]].size(0),
            self.projection_dim,
            device=embeddings[self.modality_names[0]].device,
        )

        weights = {}
        for name in self.modality_names:
            projected = self.projections[name](embeddings[name])  # (B, proj_dim)
            gated = gate_values[name] * projected  # (B, proj_dim)
            fused = fused + gated
            # Average gate value across dimensions for weight reporting
            weights[name] = gate_values[name].mean(dim=-1)  # (B,)

        fused = self.fusion_layer(fused)

        return fused, weights


class AttentionFusion(nn.Module):
    """
    Multi-head self-attention fusion across modalities.

    Treats each modality embedding as a token and applies
    self-attention to learn inter-modality relationships.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        projection_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            modality_dims: Dict mapping modality name → embedding dim.
            projection_dim: Common latent space dimension.
            num_heads: Number of attention heads.
            hidden_dim: FFN hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.projection_dim = projection_dim

        # Project each modality to common dim
        self.projections = nn.ModuleDict()
        for name, dim in sorted(modality_dims.items()):
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(projection_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.norm2 = nn.LayerNorm(projection_dim)

        # Final pooling projection
        self.pool_projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: Dict of modality name → embedding tensor (B, D_m).

        Returns:
            Tuple of:
            - fused: (B, projection_dim).
            - weights: Attention-derived modality weights (B,) per modality.
        """
        # Project each modality
        tokens = []
        for name in self.modality_names:
            projected = self.projections[name](embeddings[name])  # (B, proj_dim)
            tokens.append(projected)

        # Stack as sequence: (B, M, proj_dim)
        x = torch.stack(tokens, dim=1)

        # Self-attention with residual
        attn_out, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Mean pool across modality tokens
        fused = x.mean(dim=1)  # (B, proj_dim)
        fused = self.pool_projection(fused)

        # Extract modality weights from attention
        # attn_weights: (B, M, M) — average across heads already done by PyTorch
        weights = {}
        for i, name in enumerate(self.modality_names):
            # Sum of attention received by this modality from all others
            weights[name] = attn_weights[:, :, i].mean(dim=-1)  # (B,)

        return fused, weights


def build_fusion(
    method: str,
    modality_dims: Dict[str, int],
    config: Dict,
) -> nn.Module:
    """
    Factory function to build a fusion module.

    Args:
        method: Fusion method name ('concat', 'gated', 'attention').
        modality_dims: Dict of modality name → embedding dim.
        config: Fusion config section.

    Returns:
        Fusion module instance.
    """
    projection_dim = config.get("projection_dim", 256)
    dropout = config.get("dropout", 0.3)

    if method == "concat":
        return ConcatenationFusion(
            modality_dims=modality_dims,
            projection_dim=projection_dim,
            dropout=dropout,
        )
    elif method == "gated":
        gate_config = config.get("gated", {})
        return GatedFusion(
            modality_dims=modality_dims,
            projection_dim=projection_dim,
            dropout=dropout,
            gate_activation=gate_config.get("gate_activation", "sigmoid"),
        )
    elif method == "attention":
        attn_config = config.get("attention", {})
        return AttentionFusion(
            modality_dims=modality_dims,
            projection_dim=projection_dim,
            num_heads=attn_config.get("num_heads", 4),
            hidden_dim=attn_config.get("hidden_dim", 256),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown fusion method: '{method}'. Use concat/gated/attention.")
