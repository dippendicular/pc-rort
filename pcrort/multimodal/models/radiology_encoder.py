"""
Radiology encoder — pretrained DenseNet121 or ResNet50 backbone.

Architecture:
    Image → Backbone → Global Average Pooling → Projection → 512-dim embedding
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class RadiologyEncoder(nn.Module):
    """
    Radiology image encoder using pretrained CNN backbones.

    Exposes:
    - forward_features(x): Returns the embedding vector.
    - forward(x): Returns classification logits (if classifier head is set).
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        embedding_dim: int = 512,
        projection_dim: int = 512,
        dropout: float = 0.3,
        num_classes: Optional[int] = None,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone: Backbone architecture name.
            pretrained: Whether to use ImageNet-pretrained weights.
            embedding_dim: Intermediate embedding dim after backbone.
            projection_dim: Final projection output dim.
            dropout: Dropout rate in projection head.
            num_classes: If set, adds a classifier head.
            freeze_backbone: Whether to freeze backbone parameters.
        """
        super().__init__()
        self.backbone_name = backbone
        self.projection_dim = projection_dim

        # Build backbone
        self.backbone, backbone_out_dim = self._build_backbone(backbone, pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
        )

        # Optional classifier head
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(projection_dim, 1 if num_classes == 2 else num_classes)

    def _build_backbone(self, name: str, pretrained: bool):
        """Build and return (backbone_features, output_dim)."""
        if name == "densenet121":
            if HAS_TIMM:
                model = timm.create_model("densenet121", pretrained=pretrained, features_only=False)
                backbone = model.features
                out_dim = model.num_features
            else:
                weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
                model = tv_models.densenet121(weights=weights)
                backbone = model.features
                out_dim = model.classifier.in_features

        elif name == "resnet50":
            if HAS_TIMM:
                model = timm.create_model("resnet50", pretrained=pretrained)
                backbone = nn.Sequential(*list(model.children())[:-2])
                out_dim = model.num_features
            else:
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                model = tv_models.resnet50(weights=weights)
                backbone = nn.Sequential(*list(model.children())[:-2])
                out_dim = model.fc.in_features

        elif name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            backbone = nn.Sequential(*list(model.children())[:-2])
            out_dim = model.fc.in_features

        else:
            if HAS_TIMM:
                model = timm.create_model(name, pretrained=pretrained, features_only=False)
                backbone = nn.Sequential(*list(model.children())[:-1])
                out_dim = model.num_features
            else:
                raise ValueError(
                    f"Unknown backbone '{name}'. Install `timm` for more options."
                )

        return backbone, out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding vector from input image.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Embedding tensor (B, projection_dim).
        """
        features = self.backbone(x)

        # Handle DenseNet ReLU
        if self.backbone_name.startswith("densenet"):
            features = nn.functional.relu(features, inplace=True)

        # Global average pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)

        # Projection
        embedding = self.projection(features)
        return embedding

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional classification.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Dict with 'embedding' and optionally 'logits'.
        """
        embedding = self.forward_features(x)
        output = {"embedding": embedding}

        if self.classifier is not None:
            logits = self.classifier(embedding)
            output["logits"] = logits.squeeze(-1) if logits.shape[-1] == 1 else logits

        return output

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
