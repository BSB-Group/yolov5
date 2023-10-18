"""
SEA.AI
authors: Kevin Serrano,

Transforms used for training the horizon detection model.
Transforms are implemented using the albumentations library.
Useful links:
- https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/
- https://albumentations.ai/docs/getting_started/transforms_and_targets/
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils.augmentations16 import CLAHE, Clip, NormalizeMinMax
from horizon.utils import points_to_pitch_theta, points_to_hough


def horizon_augment_RGB(imgsz: int) -> A.Compose:
    """
    Augmentations for RGB images.

    Args:
        imgsz (int): image size
    """
    return A.Compose([
        # image transforms
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.CLAHE(p=0.05),
        A.Blur(p=0.2),

        # geometric transforms
        A.LongestMaxSize(max_size=imgsz),
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT),  # letterbox
        A.ShiftScaleRotate(p=1, shift_limit=0.1, scale_limit=0.25, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT),

        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        ToTensorV2(p=1.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def horizon_base_RGB(imgsz: int) -> A.Compose:
    """
    No augmentation, just resize and normalize.

    Args:
        imgsz (int): image size
    """
    return A.Compose([
        # geometric transforms
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT),  # letterbox

        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        ToTensorV2(p=1.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def horizon_augment_IR16bit(imgsz: int) -> A.Compose:
    """
    Augmentations for 16-bit IR images.

    Args:
        imgsz (int): image size
    """
    return A.Compose([
        # image transforms (16-bit compatible)
        Clip(p=1.0, lower_limit=(0.2, 0.25), upper_limit=(0.4, 0.45)),
        CLAHE(p=0.5, clip_limit=(3, 5), tile_grid_size=(-1, -1)),
        NormalizeMinMax(p=1.0),
        A.UnsharpMask(p=0.5, threshold=5),
        A.ToRGB(p=1.0),

        # geometric transforms
        A.LongestMaxSize(max_size=imgsz),
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT),  # letterbox
        A.ShiftScaleRotate(p=1, shift_limit=0.1, scale_limit=0.25, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT),

        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        ToTensorV2(p=1.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def horizon_base_IR16bit(imgsz: int,
                         lower_limit: float = 15000 / 65535,
                         upper_limit: float = 28000 / 65535,
                         clahe: bool = True,
                         unsharp_mask: bool = True
                         ) -> A.Compose:
    """
    No augmentation, just resize and normalize.

    Args:
        imgsz (int): image size
        lower_limit (float): lower limit for clipping
        upper_limit (float): upper limit for clipping
        clahe (bool): apply CLAHE
        unsharp_mask (bool): apply unsharp mask
    """
    return A.Compose([
        # image transforms (16-bit compatible)
        Clip(p=1.0, lower_limit=(lower_limit,) * 2, upper_limit=(upper_limit,) * 2),
        CLAHE(p=1.0 if clahe else 0.0, clip_limit=(4, 4), tile_grid_size=(-1, -1)),
        NormalizeMinMax(p=1.0),
        A.UnsharpMask(p=1.0 if unsharp_mask else 0.0, threshold=5),
        A.ToRGB(p=1.0),

        # geometric transforms
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT),  # letterbox

        # torch-related transforms
        A.Normalize(mean=0.0, std=1.0),  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        ToTensorV2(p=1.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def points_to_normalised_pitch_theta(imgsz: int):
    """
    Convert points to pitch, theta.
    """
    def _points_to_pitch_theta(points: np.ndarray,
                              image_w: int = imgsz,
                              image_h: int = imgsz,
                              ) -> np.ndarray:
        """
        Convert points to pitch, theta.
        """
        # normalize points
        points = np.array(points)
        points[:, 0] /= image_w
        points[:, 1] /= image_h

        # convert to pitch, theta
        x1, y1, x2, y2 = points.flatten()
        pitch, theta = points_to_pitch_theta(x1, y1, x2, y2)
        return np.array([pitch, theta])
    
    return _points_to_pitch_theta
