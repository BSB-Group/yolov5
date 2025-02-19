"""Custom augmentations following the Albumentations API."""

import random
from typing import Dict, Tuple

import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as F
from albumentations.core.transforms_interface import DualTransform

from albumentations.augmentations.geometric.resize import MaxSizeInitSchema

class ResizeIfNeeded(DualTransform):
    """
    Resize an image if its longest side is greater than a specified value.

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
    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: int | None = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(p=p, always_apply=always_apply)
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

    def apply_to_bbox(self, bbox: np.ndarray, **params) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoints(self, keypoints: np.ndarray, max_size: int = 1024, **params) -> np.ndarray:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        scale = min(1.0, scale)  # don't scale up
        return F.keypoints_scale(keypoints, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {"max_size": (self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size))}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")
