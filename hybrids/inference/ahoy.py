"""
AHOYv5 model for object and horizon detection.

TODO: add metadata to model file:
    - input_size
    - half
    - offset_buffer (how offset is encoded in model)
    - classes_names
"""

import os
import logging
from typing import Union
from pathlib import Path
import yaml
import numpy as np

# import pycuda.autoinit  # cuda context initialized manually in __init__

from .preprocessing import preprocess_yolo
from .postprocessing import xyxy_to_xyxyn, postprocess_ahoy
from .misc import Profile

logger = logging.getLogger(__name__)


def load_labels(label_path):
    """
    Load label mapping from file.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AHOYv5:
    """
    A
    Horizon and
    Object detection based on
    YOLOv5.
    """

    def __init__(self, model_path: str, device: int = 0) -> None:
        """
        Initialize the AHOYv5 model.

        Parameters
        ----------
        model_path : str
            Path to the model file containing the weights and architecture.
        device : int, optional
            CUDA device to use.
        """
        if Path(model_path).suffix == ".engine":
            from .infertrt import InferTRT

            self.model = InferTRT(model_path, device)
        else:
            raise ValueError("Only TensorRT engines are supported.")

        cls_map_fpath = model_path.replace(".engine", ".yaml")
        self.cls_map = (
            load_labels(cls_map_fpath) if os.path.exists(cls_map_fpath) else None
        )

        self.profiles = {
            "preprocess": Profile(),
            "inference": Profile(),
            "postprocess": Profile(),
        }

    def __call__(
        self,
        ims: np.ndarray,
        conf_thresh: float = 0.147,
        iou_thresh: float = 0.1,
        curve_fit: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Run the model pipeline on the given input image(s).

        Parameters
        ----------
        ims : np.ndarray
            The input image(s).
        conf : float, optional
            Confidence threshold for p(class). Predictions with score < conf_thresh
            are ignored.
        iou_thresh : float, optional
            Minimum IOU to be counted as a duplicate detection.
        curve_fit : bool, optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model.
        verbose : bool, optional
            If True, prints the time spent in each step of the pipeline.

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            - Predicted bounding boxes.
            - Confidence scores.
            - Class labels (ints)
        """

        orig_shape = ims.shape[-3:-1]  # (h, w)

        with self.profiles["preprocess"]:
            ims = self.preprocess(ims)

        with self.profiles["inference"]:
            output = self.model.forward(ims)

        with self.profiles["postprocess"]:
            preds = self.postprocess(
                output,
                orig_shape,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                curve_fit=curve_fit,
            )

        if verbose:
            for name, profile in self.profiles.items():
                print(f"{profile.dt * 1E3:>5.1f} ms - {name}")

        return preds

    def preprocess(self, ims: np.array) -> np.ndarray:
        """
        Transform the input image so that the model can infer from it.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return preprocess_yolo(ims, self.model.input_hw, self.model.fp16)

    def postprocess(
        self,
        output: tuple,
        orig_hw: tuple,
        conf_thresh: float,
        iou_thresh: float,
        offset_buffer: float = 0.15,
        curve_fit: bool = True,
    ) -> tuple:
        """
        Transform raw model output to application output.

        Parameters
        ----------
        output : tuple
            Raw model output.
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
        List[np.ndarray]
            Every element is a np.ndarray of shape (N, 6) where:
            - N is the number of detected bounding boxes,
            - first 4 are the coordinates of the bounding box,
            - 5 is the confidence score of the bounding box,
            - 6 is the class label of the detected bounding box.
        """

        return postprocess_ahoy(
            output,
            self.model.input_hw,
            orig_hw,
            conf_thresh,
            iou_thresh,
            offset_buffer,
            curve_fit,
        )

    def predict(self, img, conf_thresh, iou_thresh, curve_fit=True):
        """shortcut for detect() with output_mode="qa" """
        return self.detect(
            img,
            output_mode="qa",
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            curve_fit=curve_fit,
        )

    def detect(
        self,
        ims: np.ndarray,
        output_mode: Union[str, None] = None,
        conf_thresh: float = 0.147,
        iou_thresh: float = 0.1,
        curve_fit: bool = True,
    ):
        """
        Parameters
        ----------------------

        - ims: np.ndarray
            The input image(s).
        - output_mode: str, optional
            If "tf", returns format expected by Tensorflow Object Detection API::

                return {
                    "detection_boxes": bboxes,
                    "detection_scores": scores,
                    "detection_classes": classes,
                    "num_detections": len(bboxes),
                }

            If "qa", returns format expected by the QA system::

                for bbox, score, cls in zip(bboxes, scores, classes):
                    proposals.append([
                        bbox,
                        self.cls_map[cls],
                        score
                    ])
                return proposals

            Otherwise, returns (bboxes, scores, classes)

        - conf_thresh: float, optional
            Confidence threshold for p(class). Predictions with score < conf are ignored.
        - iou_thresh: float, optional
            Minimum IOU to be counted as a duplicate detection.
        - curve_fit: bool, optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model.
        """
        if isinstance(ims, np.ndarray):
            assert ims.ndim == 4, "Input image must have 3 dimensions (b, h, w, c)"
        elif isinstance(ims, (list, tuple)):
            assert all(im.ndim == 4 for im in ims), "All images must have 4 dimensions"

        orig_shape = ims[0].shape[:2]
        dets = self(ims, conf_thresh, iou_thresh, curve_fit)
        bboxes = np.array([det[:, :4] for det in dets])
        scores = np.array([det[:, 4] for det in dets])
        classes = np.array([det[:, 5] for det in dets])

        if output_mode == "tf":
            bboxes = xyxy_to_xyxyn(bboxes, orig_shape)  # normalize to [0, 1]
            bboxes = bboxes[:, [1, 0, 3, 2]]  # x1,y1,x2,y2 to y1,x1,y2,x2
            return {
                "detection_boxes": bboxes,
                "detection_scores": scores,
                "detection_classes": classes,
                "num_detections": len(bboxes),
            }

        if output_mode == "qa":
            bboxes = xyxy_to_xyxyn(bboxes, orig_shape)  # normalize to [0, 1]
            proposals = []
            for bbox, score, cls in zip(bboxes, scores, classes):
                proposals.append([bbox, self.cls_map[cls], score])
            return proposals

        return bboxes, scores, classes

    def close(self):
        self.model.close()
