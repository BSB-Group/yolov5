from typing import Sequence, Union
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple


def convert_16bit_to_8bit(im, augment=True):
    if not len(im.shape) == 2:
        # for some reason, some 16bit images have 3 channels
        im = im[:, :, 0]
    transform = get_16_to_8_transform(augment)
    return transform(image=im)['image']


def get_16_to_8_transform(augment):
    if augment:
        # meant to be used for training, randomness is important
        return A.Compose([
            # 15000/65535 = 0.228, 28000/65535 = 0.427
            Clip(p=1.0, lower_limit=(0.2, 0.25), upper_limit=(0.4, 0.45)),
            CLAHE(p=0.5, clip_limit=(3, 5), tile_grid_size=(-1, -1)),
            NormalizeMinMax(p=1.0),
            A.UnsharpMask(p=0.5, threshold=5),
            A.ToRGB(p=1.0),
        ])
    else:
        llimit = 15000/65535
        ulimit = 28000/65535
        # meant to be used for validation, deterministic
        return A.Compose([
            Clip(p=1.0, lower_limit=(llimit, llimit), upper_limit=(ulimit, ulimit)),
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

    def __init__(self, 
                 clip_limit: Union[float, Sequence[float]]=4.0, 
                 tile_grid_size: Union[float, Sequence[float]]=(8, 8), 
                 always_apply=False, 
                 p=0.5):
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
        lower_limit (float or (float, float)): lower limit value for clipping.
            If lower_limit is a single float value, the range will be (0, lower_limit). Default: (0.1, 0.2).
        upper_limit (float or (float, float)): upper limit value for clipping.
            If upper_limit is a single float value, the range will be (upper_limit, 1). Default: (0.8, 0.9).

    Targets:
        image

    Image types:
        uint8, uint16
    """

    def __init__(self, 
                 lower_limit: Union[float, Sequence[float]] = (0.1, 0.2),
                 upper_limit: Union[float, Sequence[float]] = (0.8, 0.9),
                 always_apply=False, 
                 p=0.5):
        super(Clip, self).__init__(always_apply, p)
        self.lower_limit = to_tuple(lower_limit, 0)
        self.upper_limit = to_tuple(upper_limit, 1)

    def apply(self, img, lower_limit=0.1, upper_limit=0.9, **params):
        max_val = np.iinfo(img.dtype).max
        a_min = int(lower_limit * max_val)
        a_max = int(upper_limit * max_val)
        return np.clip(img, a_min, a_max)

    def get_params(self):
        return {"lower_limit": random.uniform(self.lower_limit[0], self.lower_limit[1]),
                "upper_limit": random.uniform(self.upper_limit[0], self.upper_limit[1])}
