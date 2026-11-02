"""
Pathology encoder — ResNet50 patch encoder + Attention MIL aggregation.

Architecture:
    Patches → ResNet50 per-patch → Patch embeddings → Attention MIL → Patient embedding (512-dim)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class AttentionMIL(nn.Module):
    """
    Gated Attention-based MIL pooling.

    Computes attention weights over patch embeddings and produces
    a single patient-level representation.

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of patch embeddings.
            hidden_dim: Hidden dimension for attention computation.
            dropout: Dropout rate.
        """
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.attention_W = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted aggregation of patch embeddings.

        Args:
            x: Patch embeddings (B, N, D) where N is number of patches.
            mask: Boolean mask (B, N) — True for valid patches.

        Returns:
            Tuple of:
            - aggregated: Patient-level embedding (B, D).
            - attention_weights: Attention weights (B, N).
        """
        # Gated attention mechanism
        V = self.attention_V(x)   # (B, N, hidden)
        U = self.attention_U(x)   # (B, N, hidden)
        A = self.attention_W(V * U)  # (B, N, 1)
        A = A.squeeze(-1)            # (B, N)

        # Mask invalid patches
        if mask is not None:
            A = A.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(A, dim=-1)  # (B, N)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum
        aggregated = torch.bmm(
            attention_weights.unsqueeze(1), x
        ).squeeze(1)  # (B, D)

        return aggregated, attention_weights


class PatchEncoder(nn.Module):
    """
    Encodes individual pathology patches using a pretrained CNN.

    Processes patches in batches to handle large bags efficiently.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: int = 512,
    ):
        super().__init__()

        if backbone == "resnet50":
            if HAS_TIMM:
                model = timm.create_model("resnet50", pretrained=pretrained)
                self.features = nn.Sequential(*list(model.children())[:-1])
                out_dim = model.num_features
            else:
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                model = tv_models.resnet50(weights=weights)
                self.features = nn.Sequential(*list(model.children())[:-1])
                out_dim = model.fc.in_features
        elif backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(model.children())[:-1])
            out_dim = model.fc.in_features
        else:
            raise ValueError(f"Unsupported pathology backbone: {backbone}")

        # Projection to desired embedding dim
        self.projection = nn.Sequential(
            nn.Linear(out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode patches.

        Args:
            x: Patch images (*, C, H, W) — can be (B*N, C, H, W) or (B, N, C, H, W).

        Returns:
            Patch embeddings (*,  embedding_dim).
        """
        # Handle batched bags: (B, N, C, H, W) → (B*N, C, H, W)
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            features = self.features(x).view(B * N, -1)
            embeddings = self.projection(features)
            return embeddings.view(B, N, -1)
        else:
            features = self.features(x).view(x.size(0), -1)
            return self.projection(features)


class PathologyEncoder(nn.Module):
    """
    Complete pathology encoder: patch encoding + MIL aggregation.

    Takes a bag of patches per patient and returns a single patient-level embedding.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        patch_embedding_dim: int = 512,
        embedding_dim: int = 512,
        attention_hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            backbone: Patch encoder backbone.
            pretrained: Use pretrained weights.
            patch_embedding_dim: Dim of individual patch embeddings.
            embedding_dim: Final patient-level embedding dim.
            attention_hidden_dim: Hidden dim for attention.
            dropout: Dropout rate.
            num_classes: If set, adds classifier head.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Patch-level encoder
        self.patch_encoder = PatchEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=patch_embedding_dim,
        )

        # Attention MIL aggregation
        self.attention_mil = AttentionMIL(
            input_dim=patch_embedding_dim,
            hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )

        # Optional projection (if patch_embedding_dim != embedding_dim)
        if patch_embedding_dim != embedding_dim:
            self.final_projection = nn.Sequential(
                nn.Linear(patch_embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.GELU(),
            )
        else:
            self.final_projection = nn.Identity()

        # Optional classifier
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(
                embedding_dim, 1 if num_classes == 2 else num_classes
            )

    def forward_features(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patient-level embedding from a bag of patches.

        Args:
            patches: Patch images (B, N, C, H, W).
            mask: Valid patch mask (B, N).

        Returns:
            Tuple of:
            - embedding: Patient embedding (B, embedding_dim).
            - attention_weights: (B, N).
        """
        # Encode patches
        patch_embeddings = self.patch_encoder(patches)  # (B, N, patch_dim)

        # MIL aggregation
        aggregated, attn_weights = self.attention_mil(patch_embeddings, mask)

        # Final projection
        embedding = self.final_projection(aggregated)

        return embedding, attn_weights

    def forward(
        self,
        patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional classification.

        Args:
            patches: (B, N, C, H, W).
            mask: (B, N).

        Returns:
            Dict with 'embedding', 'attention_weights', and optionally 'logits'.
        """
        embedding, attn_weights = self.forward_features(patches, mask)

        output = {
            "embedding": embedding,
            "attention_weights": attn_weights,
        }

        if self.classifier is not None:
            logits = self.classifier(embedding)
            output["logits"] = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

        return output
