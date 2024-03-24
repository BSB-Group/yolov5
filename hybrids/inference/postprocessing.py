"""
Collection of postprocessing functions for object detection models.
Most functions are designed to work with NumPy arrays.
"""

from typing import Tuple, Sequence
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import softmax


def cxcywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (center_x, center_y, width, height) to
    (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (center_x, center_y, width, height) format

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    center_x, center_y, width, height = bboxes.T
    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    return np.array([top_left_x, top_left_y, bottom_right_x, bottom_right_y]).T


def xyxy_to_xywh(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    to (center_x, center_y, width, height).

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (top_left_x, top_left_y, width, height) format
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes.T
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    return np.array([top_left_x, top_left_y, width, height]).T


def xyxy_to_xyxyn(bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes in form of (x, y, x, y) from absolute pixels to relative
    pixels in range [0, 1] with respect to certain shape.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format
    shape : Tuple[int, int]
        shape in (height, width) to which should be normalized

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format in range [0, 1]
    """
    h, w = shape
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    return bboxes


def xyxyn_to_xyxy(bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes in form of (x, y, x, y) from relative pixels in range [0, 1]
    to absolute pixels with respect to certain shape.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format
    shape : Tuple[int, int]
        shape in (height, width) to which should be normalized

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes in (x, y, x, y) format in absolute pixels
    """
    h, w = shape
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    return bboxes


def scale_boxes(
    bboxes: np.ndarray,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    only_downscale: bool = False,
) -> np.ndarray:
    """
    Rescale bounding boxes from from_shape to to_shape

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    from_shape : tuple (2,)
        original shape of the image (height, width)
    to_shape : tuple (2,)
        target shape of the image (height, width)
    only_downscale : bool, optional
        whether to only downscale the boxes

    Returns
    -------
    np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """

    # gain  = old / new
    gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    gain = min(gain, 1) if only_downscale else gain
    # wh padding
    pad = (
        (from_shape[1] - to_shape[1] * gain) / 2,  # x padding
        (from_shape[0] - to_shape[0] * gain) / 2,
    )  # y padding

    bboxes[..., [0, 2]] -= pad[0]  # x padding
    bboxes[..., [1, 3]] -= pad[1]  # y padding
    bboxes[..., :4] /= gain
    clip_boxes(bboxes, to_shape)
    return bboxes


def clip_boxes(boxes, shape):
    """Clip boxes (xyxy) to image shape (height, width)"""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def intersection_box(target_box: np.ndarray, boxes: np.ndarray) -> list:
    """
    Calculate intersection of target boxes with all others in boxes

    Parameters
    ----------
    target_box : np.ndarray (4,)
        bounding box as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    boxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    Returns
    -------
    np.ndarray (N, 4)
        intersection bboxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    x1, y1, x2, y2 = target_box
    a1, b1, a2, b2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    xx1 = np.maximum(x1, a1)
    yy1 = np.maximum(y1, b1)
    xx2 = np.minimum(x2, a2)
    yy2 = np.minimum(y2, b2)

    return np.array([xx1, yy1, xx2, yy2])


def intersection_area(target_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    calculate intersection of target boxes with all others in boxes
    Returns:
        (np.ndarray) areas of intersection bounding boxes
    """
    xx1, yy1, xx2, yy2 = intersection_box(target_box, boxes)
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    return w * h


def iou(
    target_box: np.ndarray,
    boxes: np.ndarray,
    target_area: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """
    Calculate intersection over union of target box with all others

    Parameters
    ----------
    target_box : np.ndarray (4,)
        bounding box as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    boxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    target_area : np.ndarray (1,)
        area of target box
    areas : np.ndarray (N,)
        areas of all boxes
    """
    inter = intersection_area(target_box, boxes)
    union = target_area + areas - inter

    # avoid division w/ 0
    union[union == 0] += 1e-7

    return inter / union


def nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
    classes: np.ndarray = None,
) -> np.ndarray:
    """
    Non-Maximum-Supression (NMS) algorithm to remove overlapping bounding boxes.

    Parameters
    ----------
    bboxes : np.ndarray (N, 4)
        bounding boxes as (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    scores : np.ndarray (N,)
        confidence scores for each bounding box
    iou_thresh : float
        threshold for intersection over union
    classes : np.ndarray (N,)
        class labels for each bounding box
        if None, all boxes are treated as the same class (agnostic NMS)
    """
    bboxes = bboxes.astype(np.float32)
    x1, y1 = bboxes[..., 0], bboxes[..., 1]  # top left
    x2, y2 = bboxes[..., 2], bboxes[..., 3]  # bottom right

    areas = (x2 - x1) * (y2 - y1)  # calculate area per bbox
    order = np.argsort(scores)[::-1]  # sort by descending confidence

    keep = np.bool_(np.zeros_like(scores))  # # which boxes (idx) to keep

    while order.size > 1:
        # take current highest confidence box
        i = order[0]
        keep[i] = True

        # calculate intersection over union with all other boxes
        ovr = iou(bboxes[i], bboxes[order[1:]], areas[i], areas[order[1:]])

        # find boxes to keep since they are not overlapping enough
        idxs = np.where(ovr <= iou_thresh)[0]

        if classes:
            # only keep boxes with different classes
            idxs = np.union1d(idxs, np.where(classes[order[1:]] != classes[i]))

        # update order by removing suppressed boxes
        order = order[idxs + 1]  # +1 because we removed the first element

    return keep


def postprocess_yolo(
    model_output: np.ndarray,
    input_hw: tuple,
    orig_hw: tuple,
    conf_thresh: float,
    iou_thresh: float,
) -> np.ndarray:
    """
    Transform raw yolo-like output to bounding boxes, scores and classes.

    Parameters
    ----------
    model_output : np.ndarray (N, 4 + 1 + NC)
        Where:
        - N is the number of detected bounding boxes,
        - 4 are the coordinates of the bounding box,
        - 1 is the confidence score of the bounding box,
        - NC is the number of classes.
    input_hw : tuple
        Input image shape (height, width)
    orig_hw : tuple
        Original image shape (height, width)
    conf_thresh : float
        Confidence threshold for p(bbox)
        Predictions with score < conf are not considered as output.
    iou_thresh : float
        Minimum IOU to be counted as a duplicate detection.

    Returns
    -------
    (boxes, scores, classes) : tuple
        Where:
        - boxes are the coordinates of the detected bounding boxes in
        the format (top-left x, top-left y, bottom-right x, bottom-right y)
        - scores are the confidence scores of the detected bounding boxes.
        - classes are the class labels of the detected bounding boxes.
    """
    if model_output.shape[0] < model_output.shape[1]:
        from .misc import yolov8_to_yolov5

        model_output = yolov8_to_yolov5(model_output)

    # Filter out boxes with low confidence
    model_output = model_output[model_output[..., 4] > conf_thresh]

    # Unpack model_output
    bboxes = model_output[..., :4]
    scores = model_output[..., 4]
    class_prob = model_output[..., 5:]

    bboxes = cxcywh_to_xyxy(bboxes)

    # p(class | bbox) * p(bbox)
    class_scores = class_prob * scores[:, None]
    classes = np.argmax(class_scores, axis=1)

    # Class agnostic since classes is not used
    keep = nms(bboxes, scores, iou_thresh)
    bboxes = bboxes[keep]
    classes = classes[keep]
    scores = scores[keep]

    bboxes = scale_boxes(bboxes, input_hw, orig_hw, only_downscale=True)
    return bboxes, scores, classes


def offset_theta_to_points(
    offset: float, theta: float, w: int = 1, h: int = 1, offset_buffer: float = 0.15
):
    """
    Convert offset and theta to two points.

    Args:
        offset (float): in [0,1] (horizon line center at offset=0.5)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)

    Returns:
        (x1, y1), (x2, y2)
    """
    m, b = offset_theta_to_slope_intercept(offset, theta, offset_buffer)
    (x1, y1), (x2, y2) = slope_intercept_to_points(m, b, w, h)
    y1, y2 = h - y1, h - y2  # invert y-axis since origin is top left corner
    return (x1, y1), (x2, y2)


def offset_theta_to_slope_intercept(
    offset: float, theta: float, offset_buffer: float = 0.15
):
    """
    Convert offset and theta to slope-intercept form of line: m, b.
    """

    # "decode" from [0, 1]
    offset = (offset - offset_buffer) / (1 - 2 * offset_buffer)
    theta = theta * np.pi - 0.5 * np.pi  # rad [-pi/2, pi/2]

    m = np.tan(theta)
    b = offset - m * 0.5
    return m, b


def slope_intercept_to_points(m: float, b: float, w: int = 1, h: int = 1):
    """
    Convert slope-intercept form of line to two points.
    """

    if m == np.inf:
        x_1, y_1 = b * h, 0
        x_2, y_2 = b * h, h
    else:
        x_1, y_1 = 0, b * h
        x_2, y_2 = w, m * w + b * h
    return (x_1, y_1), (x_2, y_2)


def gaussian_curve_fit(data: np.ndarray, ftol: float = 1e-4, xtol: float = 1e-4):
    """
    Fit a gaussian to the softmax data to find peak.
    Returns the parameters of the gaussian: (A, mu, sigma)
    """

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    # Initial guess for the parameters (amplitude, mean, std deviation)
    p0 = [data.max(), data.argmax() / data.shape[-1], 0.001]
    x = np.linspace(0, 1, data.shape[0], endpoint=False)
    kwargs = {"ftol": ftol, "xtol": xtol}

    curve_params, _ = curve_fit(gaussian, x, data, p0=p0, **kwargs)
    return curve_params  # A, mu, sigma


def postprocess_offset_theta(
    offset: np.array,
    theta: np.array,
    orig_hw: tuple,
    offset_buffer: float = 0.15,
    do_curve_fit: bool = False,
):
    """
    Get line points and score from offset and theta logits.

    Parameters
    ----------
    offset : np.array
        Offset vector (classification logits)
    theta : np.array
        Theta vector (classification logits)
    orig_hw : tuple
        Original image shape (height, width)
    offset_buffer : float, optional
        Buffer for offset-theta model
    do_curve_fit : bool, optional
        Whether to fit a gaussian curve to the softmax data

    Returns
    -------
    (offset, theta): tuple
        offset = (offset, offset_score)
        theta = (theta, theta_score)
    """

    offset, theta = softmax(offset), softmax(theta)

    if do_curve_fit:
        offset_prob, offset, _ = gaussian_curve_fit(offset)
        theta_prob, theta, _ = gaussian_curve_fit(theta)
    else:
        offset_prob, offset = offset.max(), offset.argmax() / len(offset)
        theta_prob, theta = theta.max(), theta.argmax() / len(theta)

    # offset, theta = 0.6859624508938708, 0.48873562950280525
    points = offset_theta_to_points(offset, theta, offset_buffer=offset_buffer)
    points = np.array(points).flatten()
    points = scale_boxes(points, (1, 1), orig_hw)
    score = offset_prob * theta_prob  # pseudo-probability (because of softmax)

    return points, score, -1


def postprocess_ahoy(
    output: tuple,
    input_hw: tuple,
    orig_hw: tuple,
    conf_thresh: float,
    iou_thresh: float,
    offset_buffer: float = 0.15,
    do_curve_fit: bool = True,
) -> Sequence[np.ndarray]:
    """
    Transform raw model output to application output.

    Parameters
    ----------
    output : tuple
        Raw model output.
    input_hw : tuple
        Input image shape (height, width)
    orig_hw : tuple
        Original image shape (height, width)
    conf_thresh : float
        Confidence threshold for p(bbox)
        Predictions with score < conf are not considered as output.
    iou_thresh : float
        Minimum IOU to be counted as a duplicate detection.
    offset_buffer : float, optional
        Buffer for offset-theta model (only used if model has 3 outputs).

    Returns
    -------
    Sequence[np.ndarray]
        Every element is a np.ndarray of shape (N, 6) where:
        - N is the number of detected bounding boxes,
        - first 4 are the coordinates of the bounding box,
        - 5 is the confidence score of the bounding box,
        - 6 is the class label of the detected bounding box.
    """

    preds = []  # store List[x1,y1,x2,y2,conf,cls] per batch

    for det_logits in output[0]:
        bboxes, confs, classes = postprocess_yolo(
            det_logits,
            input_hw,
            orig_hw,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        preds.append(np.c_[bboxes, confs, classes])

    # iterate over offset-theta outputs grouped by batch
    for i, (offset_logits, theta_logits) in enumerate(zip(*output[1:])):
        points, conf, cls = postprocess_offset_theta(
            offset_logits, theta_logits, orig_hw, offset_buffer, do_curve_fit
        )
        # append hoizon line as a bbox (x1, y1, x2, y2)
        preds[i] = np.r_[preds[i], np.r_[points, conf, cls].reshape(1, -1)]

    return preds
