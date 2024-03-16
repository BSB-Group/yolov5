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
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Downscale the image to fit the desired size, preserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    - interpolation: interpolation method to use.
    """
    if image.shape[0] <= desired_size[1] and image.shape[1] <= desired_size[0]:
        return image
    return resize_image_keeping_aspect_ratio(
        image=image, desired_size=desired_size, interpolation=interpolation
    )


def resize_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize reserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    - interpolation: interpolation method to use.
    """
    if max(desired_size) == max(image.shape[:2]):
        return image

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
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


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
    x = np.stack([letterbox_image(im, input_hw) for im in ims], axis=0)
    x = x.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    x = x.astype(np.float32) * (1 / 255.0)

    # NOTE: converting to float16 is rather slow on numpy
    # leave it to tensorrt to do the conversion or
    # uncomment the following line to convert to float16
    x = x.astype(np.float16) if fp16 else x

    return x


def resize_and_center_images_in_batch(
    input_batch: np.ndarray, output_batch: np.ndarray, to_dtype: bool = False
):
    """
    Resize and center images from an input batch into a preallocated output batch array.

    Parameters
    ----------
    input_batch : np.ndarray
        The input batch of images with shape (N, H_in, W_in, C), where N is the batch
        size, H_in and W_in are the height and width of the images, and C is the number
        of channels.

    output_batch : np.ndarray
        The preallocated output batch array with shape (N, C, H_out, W_out), where
        H_out and W_out are the target height and width. The array's contents will
        be overwritten with the resized and centered images.

    Returns
    -------
    None
        The function modifies the output_batch in place and does not return a value.

    Notes
    -----
    - Ensure that the output_batch is preallocated with the desired data type and
      dimensions to avoid unintended behavior.
    - Significant variation in input image dimensions may lead to visual artifacts in
      the output_batch due to overlapping or incomplete coverage when pasting resized
      images.
    """

    h_o, w_o = output_batch.shape[2], output_batch.shape[3]
    input_batch = np.stack(
        [downscale_image_keeping_aspect_ratio(im, (h_o, w_o)) for im in input_batch],
        axis=0,
    )

    # Rearrange the dimensions
    input_batch = input_batch.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    input_batch = input_batch.astype(np.float32) * (1 / 255.0)

    # Calculate padding offsets for centering
    h, w = input_batch.shape[2], input_batch.shape[3]
    pad_height = (h_o - h) // 2
    pad_width = (w_o - w) // 2

    # Paste the input_batch into the output_batch
    output_batch[:, :, pad_height: pad_height + h, pad_width: pad_width + w] = (
        input_batch
    )

    return output_batch
