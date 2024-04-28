"""
Custom augmentations following the Albumentations API.
"""

from typing import Dict, Tuple, Union, Sequence
import random

import numpy as np
import cv2
from albumentations.core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
)
from albumentations.augmentations.geometric import functional as F


class ResizeIfNeeded(DualTransform):
    """Resize an image if its longest side is greater than a specified value.

    Args:
        max_size (int, list of int): maximum size of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(ResizeIfNeeded, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        **params,
    ) -> np.ndarray:
        if max(img.shape[:2]) > max_size:
            return F.longest_max_size(img, max_size=max_size, interpolation=interpolation)
        return img  # skip the transformation (don't scale up)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, max_size: int = 1024, **params
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        scale = min(1.0, scale)  # don't scale up
        return F.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {
            "max_size": (
                self.max_size
                if isinstance(self.max_size, int)
                else random.choice(self.max_size)
            )
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")
