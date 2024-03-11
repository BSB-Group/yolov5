from typing import Tuple
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
            im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp
        )
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def letterbox_image(
    image: np.ndarray,
    desired_size: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Resize and pad image to fit the desired size, preserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    - color: tuple (B, G, R) representing the color to pad with.

    Returns:
    - letterboxed image.
    """
    resized_img = resize_image_keeping_aspect_ratio(
        image=image,
        desired_size=desired_size,
    )
    new_height, new_width = resized_img.shape[:2]
    top_padding = (desired_size[1] - new_height) // 2
    bottom_padding = desired_size[1] - new_height - top_padding
    left_padding = (desired_size[0] - new_width) // 2
    right_padding = desired_size[0] - new_width - left_padding
    return cv2.copyMakeBorder(
        resized_img,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=color,
    )


def downscale_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> np.ndarray:
    if image.shape[0] <= desired_size[1] and image.shape[1] <= desired_size[0]:
        return image
    return resize_image_keeping_aspect_ratio(image=image, desired_size=desired_size)


def resize_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize reserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    """
    img_ratio = image.shape[1] / image.shape[0]
    desired_ratio = desired_size[0] / desired_size[1]

    # Determine the new dimensions
    if img_ratio >= desired_ratio:
        # Resize by width
        new_width = desired_size[0]
        new_height = int(desired_size[0] / img_ratio)
    else:
        # Resize by height
        new_height = desired_size[1]
        new_width = int(desired_size[1] * img_ratio)

    # Resize the image to new dimensions
    return cv2.resize(image, (new_width, new_height))


def preprocess_yolo(
    ims: np.array, input_hw: Tuple[int, int], fp16: bool = False
) -> np.ndarray:
    """
    Transform the input image so that the engine can infer from it.

    Parameters
    ----------
    ims : np.ndarray
        The input image(s)
    fp16 : bool, optional
        If True, the image is transformed to float16. Otherwise, float32.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    ims = np.expand_dims(ims, axis=0) if ims.ndim == 3 else ims
    B, H, W, C = ims.shape
    x = letterbox_image(
        ims.transpose(1, 2, 0, 3).reshape(H, W, B * C),  # letterbox all at once
        desired_size=input_hw[::-1],
    )
    x = x.reshape(*input_hw, B, C).transpose((2, 3, 0, 1))  # BHWC to BCHW
    x = x.astype(np.float32) * (1 / 255.0)
    
    # NOTE: converting to float16 is rather slow on numpy
    # leave it to tensorrt to do the conversion or
    # uncomment the following line to convert to float16
    # x = x.astype(np.float16) if fp16 else x
    
    return x
