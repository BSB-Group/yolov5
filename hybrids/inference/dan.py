"""
DANv5 model for day and night detection.

TODO: add metadata to model file:
    - input_size
    - half
    - offset_buffer (how offset is encoded in model)
    - classes_names
"""

import os
import logging
from typing import Union, Sequence, List, Dict
from pathlib import Path
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder

# import pycuda.autoinit  # cuda context initialized manually in __init__

from .preprocessing import preprocess_yolo, resize_and_center_images_in_batch
from .postprocessing import postprocess_ahoy
from .misc import Profile
from .ahoy import AHOYv5

logger = logging.getLogger(__name__)


def load_labels(label_path):
    """
    Load label mapping from file.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DANv5:
    """
    Day
    And
    Night detection model

    Based on two AHOYv5 models.
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

        cls_maps_fpath = model_path.replace(".engine", ".yaml")
        self.cls_maps = None
        self.label_encoder = None
        if os.path.exists(cls_maps_fpath):
            self.cls_maps = load_labels(cls_maps_fpath)
            if isinstance(self.cls_maps, list):
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(
                    list({v for cls_map in self.cls_maps for v in cls_map.values()})
                )
            else:
                # assume both outputs have same mapping
                self.cls_maps = [self.cls_maps] * len(self.model.inputs)
                self.label_encoder = None
            logger.info(f"{self.cls_maps}")

        self.profiles = {
            "preprocess": Profile(),
            "inference": Profile(),
            "postprocess": Profile(),
        }

    @staticmethod
    def get_label2id_map(cls_maps: List[Dict]):
        """Union of two class maps."""
        # get both class maps
        i0_cls_map, i1_cls_map = cls_maps
        # remove ids < 0
        i0_cls_map = {k: v for k, v in i0_cls_map.items() if v >= 0}
        i1_cls_map = {k: v for k, v in i1_cls_map.items() if v >= 0}
        # get union of both class maps
        unique_classes = set(i0_cls_map.values()) | set(i1_cls_map.values())
        # create a common class map
        return {cls: i for i, cls in enumerate(sorted(unique_classes))}

    def __call__(
        self,
        ims: Sequence[np.ndarray],
        conf_thresh: Union[float, Sequence[float]] = 0.147,
        iou_thresh: Union[float, Sequence[float]] = 0.1,
        do_curve_fit: Union[bool, Sequence[bool]] = True,
        verbose: bool = False,
    ) -> Sequence[Sequence[np.ndarray]]:
        """
        Run the entire model pipeline on the given input(s) image(s).

        Parameters
        ----------
        img: Sequence[np.ndarray]
            The input image(s). If list, each element is an input to a multi-input
            model.
        conf : float or Sequence[float], optional
            Confidence threshold for p(class). Predictions with score < conf_thresh
            are ignored. If single value, applies to all inputs.
        iou_thresh : float or Sequence[float], optional
            Minimum IOU to be counted as a duplicate detection. If single value,
            applies to all inputs.
        do_curve_fit : bool or Sequence[bool], optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model. If single value, applies to all
            inputs.
        verbose : bool, optional
            If True, prints the time spent in each step of the pipeline.

        Returns
        -------
        Sequence[Sequence[np.ndarray]]
            Outer sequence corresponds to each model's batch input.
            Inner sequence corresponds to each sample in the batch.
            Arrays are of size (N, 6), where N is the number of detected
            bounding boxes, and 6 are the [x1,y1,x2,y2,score,class] per detection.
        """

        orig_shapes = [im.shape[-3:-1] for im in ims]  # (h, w)

        with self.profiles["preprocess"]:
            ims = self.preprocess(ims)

        with self.profiles["inference"]:
            output = self.model.forward(ims)

        with self.profiles["postprocess"]:
            preds = self.postprocess(
                output,
                orig_shapes,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                do_curve_fit=do_curve_fit,
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
        return self.preprocess_fast(ims)

    def preprocess_yolo(self, ims: Sequence[np.array]) -> Sequence[np.ndarray]:
        """
        Transform the input image so that the model can infer from it.

        Parameters
        ----------
        ims : list[np.ndarray]
            The input(s) image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return [
            preprocess_yolo(im, input_hw, self.model.fp16)
            for input_hw, im in zip(self.model.input_hw, ims)
        ]

    def preprocess_fast(self, ims: Sequence[np.array]) -> Sequence[np.ndarray]:
        """
        Transform the input image so that the model can infer from it.
        This runs under the assumption that input images have a constant size
        throughout the pipeline.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        return [
            resize_and_center_images_in_batch(im, alloc_array)
            for im, alloc_array in zip(ims, self.model.alloc_arrays)
        ]

    def postprocess(
        self,
        outputs: tuple,
        orig_hws: Sequence[tuple],
        conf_thresh: Union[float, Sequence[float]],
        iou_thresh: Union[float, Sequence[float]],
        offset_buffer: Union[float, Sequence[float]] = 0.15,
        do_curve_fit: Union[bool, Sequence[bool]] = True,
    ) -> Sequence[Sequence[np.ndarray]]:
        """
        Transform raw model output to application output.

        Parameters
        ----------
        outputs : tuple
            Raw model output.
        orig_hws : tuple or Sequence[tuple]
            Original image shape (height, width)
        conf_thresh : float or Sequence[float]
            Confidence threshold for p(bbox)
            Predictions with score < conf are not considered as output.
            If single value, applies to all inputs.
        iou_thresh : float or Sequence[float]
            Minimum IOU to be counted as a duplicate detection.
            If single value, applies to all inputs.
        offset_buffer : float or Sequence[float]
            Buffer for offset-theta model. If single value, applies to all inputs.
        do_curve_fit : bool or Sequence[bool]
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model. If single value, applies to all
            inputs.

        Returns
        -------
        Sequence[Sequence[np.ndarray]]
            Outer sequence corresponds to each model's batch input.
            Inner sequence corresponds to each sample in the batch.
            Arrays are of size (N, 6), where N is the number of detected
            bounding boxes, and 6 are the [x1,y1,x2,y2,score,class] per detection.
        """
        n = len(outputs) // len(self.model.inputs)
        grouped_outputs = [outputs[i : i + 3] for i in range(0, len(outputs), n)]

        if not isinstance(conf_thresh, list):
            conf_thresh = [conf_thresh] * len(grouped_outputs)
        if not isinstance(iou_thresh, list):
            iou_thresh = [iou_thresh] * len(grouped_outputs)
        if not isinstance(offset_buffer, list):
            offset_buffer = [offset_buffer] * len(grouped_outputs)
        if not isinstance(do_curve_fit, list):
            do_curve_fit = [do_curve_fit] * len(grouped_outputs)

        return [
            postprocess_ahoy(*args)
            for args in zip(
                grouped_outputs,
                self.model.input_hw,
                orig_hws,
                conf_thresh,
                iou_thresh,
                offset_buffer,
                do_curve_fit,
            )
        ]

    def predict(self, img, conf_thresh, iou_thresh, do_curve_fit=True):
        """shortcut for detect() with output_mode="qa" """
        return self.detect(
            img,
            output_mode="qa",
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            do_curve_fit=do_curve_fit,
        )

    def detect(
        self,
        ims: np.ndarray,
        output_mode: Union[str, None] = "qa",
        conf_thresh: float = 0.147,
        iou_thresh: float = 0.1,
        do_curve_fit: bool = False,
    ):
        """
        Parameters
        ----------------------

        - ims: np.ndarray
            The input image(s).
         - output_mode: str, default "qa"
            If "tf", returns format expected by Tensorflow Object Detection API::

                return {
                    "detection_boxes": bboxes,
                    "detection_scores": scores,
                    "detection_classes": classes,
                    "num_detections": len(bboxes),
                }

            If "qa", returns format expected by the QA system::

                return [[[x1, y1, x2, y2], class_name, class_id, score], ...]

            Otherwise, returns (bboxes, scores, classes)

        - conf_thresh: float, optional
            Confidence threshold for p(class). Predictions with score<conf are ignored.
        - iou_thresh: float, optional
            Minimum IOU to be counted as a duplicate detection.
        - do_curve_fit: bool, optional
            If True, uses curve fitting to get the offset and theta values.
            Applicable only for offset-theta model.
        """
        if isinstance(ims, np.ndarray):
            assert ims.ndim == 4, "Input image must have 3 dimensions (b, h, w, c)"
        elif isinstance(ims, (list, tuple)):
            assert all(im.ndim == 4 for im in ims), "All images must have 4 dimensions"

        orig_shapes = [im.shape[-3:-1] for im in ims]
        outputs_dets = self(ims, conf_thresh, iou_thresh, do_curve_fit)
        outputs_dets = [
            AHOYv5.to_mode(dets, output_mode, orig_shape, cls_map)
            for dets, orig_shape, cls_map in zip(
                outputs_dets, orig_shapes, self.cls_maps
            )
        ]

        if output_mode == "qa" and self.label_encoder is not None:
            for output_dets in outputs_dets:  # for each batched model output
                for output in output_dets:  # for each item in batch
                    for det in output:  # for each detection
                        det[2] = self.label_encoder.transform([det[1]]).item()

        return outputs_dets

    def close(self):
        """Run this before exiting the program to free up resources."""
        self.model.close()
