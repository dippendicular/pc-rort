"""Radiology image preprocessing and augmentation transforms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision import transforms


def get_radiology_transforms(
    config: Dict[str, Any],
    is_training: bool = True,
) -> transforms.Compose:
    """
    Build radiology image transforms from config.

    Args:
        config: Radiology config section (config['radiology']).
        is_training: Whether to include augmentation.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    image_size = config.get("image_size", 224)
    normalize_mean = config.get("normalize_mean", [0.485, 0.456, 0.406])
    normalize_std = config.get("normalize_std", [0.229, 0.224, 0.225])
    aug_config = config.get("augmentation", {})

    transform_list = []

    if is_training and aug_config:
        # Random resized crop
        if aug_config.get("random_resized_crop", False):
            crop_scale = aug_config.get("crop_scale", [0.8, 1.0])
            transform_list.append(
                transforms.RandomResizedCrop(
                    image_size, scale=tuple(crop_scale)
                )
            )
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))

        # Random flips
        if aug_config.get("random_horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if aug_config.get("random_vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        # Random rotation
        rotation = aug_config.get("random_rotation", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        # Color jitter
        cj = aug_config.get("color_jitter", {})
        if cj:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                    hue=cj.get("hue", 0),
                )
            )
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    return transforms.Compose(transform_list)
