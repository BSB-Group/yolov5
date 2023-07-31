import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple


def convert_16bit_to_8bit(im, augment=True):
    transform = get_16_to_8_transform(augment)
    return transform(image=im)['image']


def get_16_to_8_transform(augment):
    if augment:
        # meant to be used for training, randomness is important
        return A.Compose([
            CLAHE(p=0.5, clip_limit=(3, 5), tile_grid_size=(-1, -1)),
            Clip(p=1.0, max_span=(0.15, 0.25)),
            NormalizeMinMax(p=1.0),
            A.ToRGB(p=1.0),
        ])
    else:
        # meant to be used for validation, deterministic
        return A.Compose([
            Clip(p=1.0, max_span=(0.2, 0.2)),
            NormalizeMinMax(p=1.0),
            A.ToRGB(p=1.0),
        ])


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
            If (-1, -1), optimal value will be calculated based on image size.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CLAHE, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        if self.tile_grid_size == (-1, -1):
            # compute tile_grid_size based on image size
            tile_grid_size = (round(max(img.shape) / 160), ) * 2
        else:
            tile_grid_size = self.tile_grid_size
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


class NormalizeMinMax(ImageOnlyTransform):
    """Normalize image to 0-255 range using min-max scaling.

    Targets:
        image

    Image types:
        uint8, uint16
    """

    def __init__(self, always_apply=False, p=0.5):
        super(NormalizeMinMax, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


class Clip(ImageOnlyTransform):
    """Clip image to a certain range.

    Args:
        max_contrast (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (max_contrast, 1). 
            Default: (0.8, 1).

    Targets:
        image

    Image types:
        uint8, uint16
    """

    def __init__(self, max_span=0.8, always_apply=False, p=0.5):
        super(Clip, self).__init__(always_apply, p)
        self.max_span = to_tuple(max_span, 1)

    def apply(self, img, max_span=0.8, **params):
        span = int(max_span * np.iinfo(img.dtype).max)
        img_min = np.min(img)
        img_max = min(np.max(img), img_min + span)
        return np.clip(img, img_min, img_max)

    def get_params(self):
        return {"max_span": random.uniform(self.max_span[0], self.max_span[1])}
