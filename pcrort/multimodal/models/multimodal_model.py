"""
Multimodal model — combines modality encoders, fusion, and classifier.

Returns: (logits, embeddings, fusion_weights)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .radiology_encoder import RadiologyEncoder
from .pathology_encoder import PathologyEncoder
from .genomics_encoder import GenomicsEncoder
from .fusion import build_fusion
from .classifier import ClassifierHead
from pcrort.deliberation import PCRORTDeliberation
from pcrort.selective import SelectivePolicy


class MultimodalModel(nn.Module):
    """
    End-to-end multimodal cancer detection model.

    Combines:
    - Radiology encoder (optional)
    - Pathology encoder (optional)
    - Genomics encoder (optional)
    - Multimodal fusion
    - Classification head

    At least two modalities must be enabled.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        genomics_input_dim: Optional[int] = None,
    ):
        """
        Args:
            config: Full merged configuration dict.
            genomics_input_dim: Number of genomics features (after preprocessing).
        """
        super().__init__()

        modalities_cfg = config.get("modalities", {})
        self.use_radiology = modalities_cfg.get("use_radiology", True)
        self.use_pathology = modalities_cfg.get("use_pathology", True)
        self.use_genomics = modalities_cfg.get("use_genomics", True)

        fusion_cfg = config.get("fusion", {})
        encoder_dims = config.get("encoder_dims", {})
        classifier_cfg = config.get("classifier", {})
        dataset_cfg = config.get("dataset", {})
        num_classes = dataset_cfg.get("num_classes", 2)

        projection_dim = fusion_cfg.get("projection_dim", 256)
        pcrort_cfg = config.get("pcrort", {})
        self.pcrort_enabled = pcrort_cfg.get("enabled", False)

        # Build modality dims dict for active modalities
        modality_dims: Dict[str, int] = {}

        # --- Radiology Encoder ---
        self.radiology_encoder = None
        if self.use_radiology:
            rad_cfg = config.get("radiology", {})
            rad_proj_dim = encoder_dims.get("radiology", 512)
            self.radiology_encoder = RadiologyEncoder(
                backbone=rad_cfg.get("backbone", "densenet121"),
                pretrained=rad_cfg.get("pretrained", True),
                embedding_dim=rad_cfg.get("embedding_dim", 512),
                projection_dim=rad_proj_dim,
                dropout=rad_cfg.get("dropout", 0.3),
                num_classes=None,  # No individual classifier in multimodal
                freeze_backbone=rad_cfg.get("freeze_backbone", False),
            )
            modality_dims["radiology"] = rad_proj_dim

        # --- Pathology Encoder ---
        self.pathology_encoder = None
        if self.use_pathology:
            path_cfg = config.get("pathology", {})
            path_emb_dim = encoder_dims.get("pathology", 512)
            self.pathology_encoder = PathologyEncoder(
                backbone=path_cfg.get("backbone", "resnet50"),
                pretrained=path_cfg.get("pretrained", True),
                patch_embedding_dim=path_cfg.get("patch_embedding_dim", 512),
                embedding_dim=path_emb_dim,
                attention_hidden_dim=path_cfg.get("attention_hidden_dim", 256),
                dropout=path_cfg.get("dropout", 0.3),
                num_classes=None,
            )
            modality_dims["pathology"] = path_emb_dim

        # --- Genomics Encoder ---
        self.genomics_encoder = None
        if self.use_genomics:
            gen_cfg = config.get("genomics", {})
            gen_emb_dim = encoder_dims.get("genomics", 256)

            if genomics_input_dim is None:
                genomics_input_dim = gen_cfg.get("input_dim")
            if genomics_input_dim is None:
                raise ValueError(
                    "Genomics input_dim must be provided either in config or "
                    "as genomics_input_dim argument."
                )

            self.genomics_encoder = GenomicsEncoder(
                input_dim=genomics_input_dim,
                hidden_dims=gen_cfg.get("hidden_dims", [1024, 512, 256]),
                embedding_dim=gen_emb_dim,
                dropout=gen_cfg.get("dropout", 0.3),
                activation=gen_cfg.get("activation", "gelu"),
                normalization=gen_cfg.get("normalization", "layernorm"),
                num_classes=None,
            )
            modality_dims["genomics"] = gen_emb_dim

        if len(modality_dims) < 1:
            raise ValueError("At least one modality must be enabled.")

        # --- Modality Projections to Common Space ---
        self.modality_projections = nn.ModuleDict()
        for name, dim in modality_dims.items():
            if dim != projection_dim:
                self.modality_projections[name] = nn.Sequential(
                    nn.Linear(dim, projection_dim),
                    nn.BatchNorm1d(projection_dim),
                    nn.GELU(),
                )
            else:
                self.modality_projections[name] = nn.Identity()

        # Update dims for fusion
        projected_dims = {name: projection_dim for name in modality_dims}

        # --- Fusion ---
        if len(modality_dims) > 1:
            self.fusion = build_fusion(
                method=fusion_cfg.get("method", "gated"),
                modality_dims=projected_dims,
                config=fusion_cfg,
            )
        else:
            self.fusion = None

        # --- Classifier ---
        self.classifier = ClassifierHead(
            input_dim=projection_dim,
            num_classes=num_classes,
            hidden_dims=classifier_cfg.get("hidden_dims", [256, 128]),
            dropout=classifier_cfg.get("dropout", 0.3),
        )
        self.deliberation = PCRORTDeliberation(
            dim=projection_dim,
            heads=pcrort_cfg.get("relation_heads", 4),
            dropout=fusion_cfg.get("dropout", 0.3),
        ) if self.pcrort_enabled else None
        self.selective_policy = SelectivePolicy(
            acceptance_threshold=pcrort_cfg.get("acceptance_threshold", .8),
            challenge_threshold=pcrort_cfg.get("challenge_threshold", .2),
            alpha=pcrort_cfg.get("conformal_alpha", .1),
        ) if self.pcrort_enabled else None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through all modalities, fusion, and classifier.

        Args:
            batch: Dict from the dataloader containing:
                - radiology_image: (B, C, H, W)
                - pathology_patches: (B, N, C, H, W)
                - pathology_mask: (B, N)
                - genomics_expression: (B, D_g)
                - label: (B,)

        Returns:
            Dict with:
            - logits: Classification logits
            - embeddings: Dict of per-modality embeddings
            - fused_embedding: Fused representation
            - fusion_weights: Modality contribution weights
        """
        embeddings: Dict[str, torch.Tensor] = {}

        # Encode radiology
        if self.use_radiology and "radiology_image" in batch:
            rad_out = self.radiology_encoder.forward_features(batch["radiology_image"])
            embeddings["radiology"] = rad_out

        # Encode pathology
        if self.use_pathology and "pathology_patches" in batch:
            path_emb, path_attn = self.pathology_encoder.forward_features(
                batch["pathology_patches"],
                mask=batch.get("pathology_mask"),
            )
            embeddings["pathology"] = path_emb

        # Encode genomics
        if self.use_genomics and "genomics_expression" in batch:
            gen_out = self.genomics_encoder.forward_features(batch["genomics_expression"])
            embeddings["genomics"] = gen_out

        # Project to common space
        projected = {}
        for name, emb in embeddings.items():
            projected[name] = self.modality_projections[name](emb)

        # PC-RORT deliberation uses only declared, observed source tokens.  The
        # availability mask is a technical bridge; clinical contracts must
        # still be admitted by EvidenceContract before calling this model.
        fusion_weights: Dict[str, torch.Tensor] = {}
        pcrort_output = {}
        if self.deliberation is not None and len(projected) > 1:
            names = sorted(projected)
            tokens = torch.stack([projected[n] for n in names], dim=1)
            available = []
            for name in names:
                available.append(batch.get(f"{name}_available", torch.ones(tokens.size(0), dtype=torch.bool, device=tokens.device)).to(tokens.device))
            available = torch.stack(available, dim=1)
            comparable = available.unsqueeze(1) & available.unsqueeze(2)
            comparable &= ~torch.eye(len(names), dtype=torch.bool, device=tokens.device).unsqueeze(0)
            confidence = available.float() * 0.75
            uncertainty = (1 - available.float())
            pcrort_output = self.deliberation(tokens, confidence, uncertainty, comparable)
            fused = pcrort_output["embedding"]
            fusion_weights = {name: available[:, i].float() for i, name in enumerate(names)}
        elif self.fusion is not None and len(projected) > 1:
            fused, fusion_weights = self.fusion(projected)
        elif len(projected) == 1:
            name = list(projected.keys())[0]
            fused = projected[name]
            fusion_weights = {name: torch.ones(fused.size(0), device=fused.device)}
        else:
            raise RuntimeError("No modality embeddings available for fusion.")

        # Classify
        logits = self.classifier(fused)

        if self.selective_policy is not None:
            coverage = torch.stack(list(fusion_weights.values()), dim=1).mean(1) if fusion_weights else torch.ones_like(logits)
            pcrort_output["selective"] = self.selective_policy.decide(
                logits, pcrort_output["challenge"], coverage,
                gap_sensitivity=1 - coverage, requestable=coverage < 1,
            )

        return {
            "logits": logits,
            "embeddings": embeddings,
            "fused_embedding": fused,
            "fusion_weights": fusion_weights,
            "pcrort": pcrort_output,
        }
