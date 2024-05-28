import contextlib
import time

import numpy as np


class Profile(contextlib.ContextDecorator):
    """Usage: @Profile() decorator or 'with Profile():' context manager"""

    def __init__(self, t=0.0):
        self.t = t
        self.start = 0.0
        self.dt = 0.0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        return time.time()


def yolov8_to_yolov5(model_output):
    """
    Convert YOLOv8 output to YOLOv5 format.

    Parameters
    ----------
    model_output : np.ndarray (4 + NC, N)
        Where:
        - N is the number of detected bounding boxes,
        - 4 are the coordinates of the bounding box,
        - NC is the number of classes.

    Returns
    -------
    np.ndarray (N, 4 + 1 + NC)
        Where:
        - N is the number of detected bounding boxes,
        - 4 are the coordinates of the bounding box,
        - 1 is the confidence score of the bounding box,
        - NC is the number of classes.
    """
    # (NC + 4, N) -> (N, NC + 4)
    model_output = np.moveaxis(model_output, 0, 1)

    bboxes = model_output[..., :4]
    class_prob = model_output[..., 4:]
    scores = np.max(class_prob, axis=1, keepdims=True)
    return np.concatenate([bboxes, scores, class_prob], axis=1)
