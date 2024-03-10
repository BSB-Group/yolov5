import time
import contextlib
import json
import numpy as np


def transform_bboxes(preditcion: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height = np.copy(preditcion[:, :4]).T
    preditcion[:, 0] = center_x - width / 2
    preditcion[:, 1] = center_y - height / 2
    preditcion[:, 2] = center_x + width / 2
    preditcion[:, 3] = center_y + height / 2
    return preditcion


class Yolo2TFLabel:
    def __init__(self, tf_category_idx: dict, yolo_label_path: str) -> None:
        """
        Initialize the Yolo2TFLabel conversion object.

        Parameters
        ----------
        tf_category_idx : dict
            Tensorflow category index dictionary.
            Contains pairs in form (i: {"name": label_str, "id": i}).
        yolo_label_path: str
            Path to dictionary containing the label mapping of the Yolo model.
            Contains pairs in form (label_nr: label_str)
        """
        tf_category_idx_rev = {s["name"]: n for n, s in tf_category_idx.items()}
        with open(yolo_label_path) as f:
            yolo_cls_map = json.load(f)  # pairs of (label_nr: label_str)
            yolo2tf = {
                int(det): tf_category_idx_rev[label]
                for det, label in yolo_cls_map.items()
            }
            self.yolo2tf_lookup = np.vectorize(yolo2tf.get)

    def __call__(self, class_pred: np.ndarray) -> np.ndarray:
        """
        Map numeric class labels from Yolo output to values as used in category index.

        Parameters
        ----------
        class_pred : np.ndarray
            Numeric class values.

        Returns
        -------
        np.ndarray
            Mapped numeric class values.
        """
        return self.yolo2tf_lookup(class_pred)


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
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
    """Convert YOLOv8 output to YOLOv5 format.

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
