import math
import numpy as np
import cv2


def resize(im: np.ndarray, max_side: int, fast: bool = True) -> tuple:
    """
    Resize an image to a maximum side length while maintaining its aspect ratio.

    Parameters
    ----------
    im : np.ndarray
        The input image as a NumPy array.
    max_side : int
        The maximum side length of the output image.
    fast : bool, optional
        If True, use linear interpolation. Otherwise, use area interpolation.

    Returns
    -------
    (im, (h0, w0), (h_resized, w_resized))
    """
    h0, w0 = im.shape[:2]  # orig hw
    r = max_side / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (fast or r > 1) else cv2.INTER_AREA
        im = cv2.resize(
            im, (math.ceil(w0 * r), math.ceil(h0 * r)), 
            interpolation=interp)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def letterbox(im: np.ndarray, 
              target_shape: tuple = (1280, 1280), 
              color: tuple =(0, 0, 0),
              scaleFill=False):
    """
    Letterbox an image to a target shape. A
    Add padding to the top/bottom or left/right of the image to make it 
    the target shape.

    Parameters
    ----------
    im : np.ndarray
        The input image as a NumPy array.
    target_shape : tuple, optional
        The target shape of the image (height, width).
    color : tuple, optional
        The color of the padding.
    scaleFill : bool, optional
        If True, stretch the image to fit the target shape. Otherwise,
        pad the image with the color.

    Returns
    -------
    (im, ratio, (dw, dh))
        im : np.ndarray
            The letterboxed image.
        ratio : float
            width, height ratios
        (dw, dh) : tuple
            The padding added to the image (width, height).
    """

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)

    # Scale ratio (new / old)
    r = min(target_shape[0] / shape[0], target_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_shape = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_shape[1] - new_shape[0], target_shape[0] - new_shape[1]  # wh padding
    if scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_shape = (target_shape[1], target_shape[0])
        ratio = target_shape[1] / shape[1], target_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_shape:  # resize
        im = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)
