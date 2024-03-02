"""
Class to run TensorRT model for object and horizon detection (if available).

TODO: add metadata to engine file:
    - input_size
    - half
    - offset_buffer (how offset is encoded in model)
    - classes_names
"""

import os
import yaml
import logging
from typing import Union

import threading
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import softmax

import tensorrt as trt
import pycuda.driver as cuda

# import pycuda.autoinit  # cuda context initialized manually in __init__

from . import engine as engine_utils
from .preprocessing import letterbox
from .postprocessing import (
    cxcywh_to_xyxy,
    xyxy_to_xyxyn,
    scale_boxes,
    nms,
    gaussian_curve_fit,
    offset_theta_to_points,
)
from .misc import Profile

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logger = logging.getLogger(__name__)


def load_labels(label_path):
    """
    Load label mapping from file.
    """
    with open(label_path, "r") as f:
        class_map = yaml.safe_load(f)
    return class_map


@DeprecationWarning
class ModelData:
    # Name of input node
    INPUT_NAME = "images"
    # CHW format of model input
    INPUT_SHAPE = (3, 1280, 1280)
    # Name of output node
    OUTPUT_NAME = "output0"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]


class YoloTRT:
    """
    Object Detection with ".engine" model
    """

    def __init__(
        self, trt_engine_path: str, img_size: int = 1280, device: int = 0
    ) -> None:
        """
        Initialize the YoloTRT object.

        Parameters
        ----------
        trt_engine_path : str
            Path to the TensorRT engine file.
        img_size : int, optional
            Size of the input image.
        device : int, optional
            CUDA device to use.
        """
        self.img_size = img_size
        cls_map_fpath = trt_engine_path.replace(".engine", ".yaml")
        self.cls_map = (
            load_labels(cls_map_fpath) if os.path.exists(cls_map_fpath) else None
        )

        cuda.init()  # needed otherwise: cuDeviceGet failed: initialization error
        self.ctx = cuda.Device(device).make_context()

        # Display requested engine settings to stdout
        print(f"Loading TensorRT inference engine {trt_engine_path}")
        self.trt_engine = engine_utils.load_engine(trt_engine_path, TRT_LOGGER)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = (
            engine_utils.allocate_buffers(self.trt_engine)
        )

        self.batch_size = self.inputs[0].shape[0]
        self.fp16 = self.inputs[0].dtype == np.float16
        print(f" - Input shape: {self.inputs[0].shape}")
        print(f" - Input dtype: {self.inputs[0].dtype}")

        # # Allocate memory for multiple usage [e.g. multiple batch inference]
        # input_volume = trt.volume(ModelData.INPUT_SHAPE)
        # self.numpy_array = np.zeros(
        #     (self.trt_engine.max_batch_size, input_volume))

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
        Run the entire TensorRT engine pipeline on the given input image.

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

        orig_shape = ims.shape[-3:-1]

        with self.profiles["preprocess"]:
            ims = self.preprocess(ims)

        with self.profiles["inference"]:
            output = self.forward(ims)

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
            assert ims.ndim == 3, "Input image must have 3 dimensions (h, w, c)"
        elif isinstance(ims, (list, tuple)):
            assert all(im.ndim == 3 for im in ims), "All images must have 3 dimensions"
            
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

    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):

        # Transfer input data to the GPU.
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)

        # Run inference.
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )

        # Transfer predictions back from the GPU.
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def forward(self, ims: np.ndarray) -> np.ndarray:
        """
        Fetch the raw predictions from the network for the given input image.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s).

        Returns
        -------
        np.ndarray
            Raw predictions.
        """
        s = self.inputs[0].shape
        assert ims.shape == s, f"input size {ims.shape} not equal to model size {s}"

        # NOTE: when using multiple threads, make sure to handle CUDA context
        # https://forums.developer.nvidia.com/t/how-to-use-tensorrt-by-the-multi-threading-package-of-python/123085/8
        threading.Thread.__init__(self)
        self.ctx.push()

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, ims.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        # inference_start_time = time.time()

        # Fetch output from the model
        detection_outs = self.do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )

        # Reshape outputs
        detection_outs = [
            np.reshape(det, out.shape) for det, out in zip(detection_outs, self.outputs)
        ]

        # Output inference time
        # print("TensorRT inference time: {} ms".format(
        #     int(round((time.time() - inference_start_time) * 1000))))

        self.ctx.pop()

        # And return results
        return detection_outs

    def preprocess(self, ims: np.array) -> np.ndarray:
        """
        Transform the input image so that the engine can infer from it.

        Parameters
        ----------
        ims : np.ndarray
            The input image(s)
        half : bool, optional
            If True, the image is transformed to float16. Otherwise, float32.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """

        ims = np.expand_dims(ims, axis=0) if ims.ndim == 3 else ims
        x = [letterbox(im, target_shape=(self.img_size,) * 2)[0] for im in ims]  # pad
        x = np.ascontiguousarray(
            np.array(x).transpose((0, 3, 1, 2))
        )  # stack and BHWC to BCHW

        # Convert to float and normalize
        # NOTE: converting first to float16 is slower!
        x = x.astype(np.float32) * (1 / 255.0)
        if self.fp16:
            x = x.astype(np.float16)

        return x

    def postprocess(
        self,
        output: tuple,
        orig_shape: tuple,
        conf_thresh: float,
        iou_thresh: float,
        offset_buffer: float = 0.15,
        curve_fit: bool = True,
    ) -> tuple:
        """
        Transform raw engine output to application output.

        Parameters
        ----------
        output : tuple
            Raw engine output.
        orig_shape : tuple
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
        (boxes, scores, classes) : tuple
            Where:
            - boxes are the coordinates of the detected bounding boxes in
            the format (top-left x, top-left y, bottom-right x, bottom-right y)
            - scores are the confidence scores of the detected bounding boxes.
            - classes are the class labels of the detected bounding boxes.
        """

        preds = []  # store List[x1,y1,x2,y2,conf,cls] per batch

        for det_logits in output[0]:
            bboxes, confs, classes = self.postprocess_bboxes(
                det_logits,
                orig_shape,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
            )
            preds.append(np.c_[bboxes, confs, classes])

        if len(output) == 3:
            for i, (offset_logits, theta_logits) in enumerate(zip(*output[1:])):
                points, conf, cls = self.postprocess_offset_theta(
                    offset_logits, theta_logits, orig_shape, offset_buffer, curve_fit
                )
                # append hoizon line as a bbox (x1, y1, x2, y2)
                row = np.r_[points, conf, cls].reshape(1, -1)
                preds[i] = np.r_[preds[i], row]

        return preds

    def postprocess_bboxes(
        self,
        model_output: np.ndarray,
        orig_shape: tuple,
        conf_thresh: float,
        iou_thresh: float,
    ) -> np.ndarray:
        """
        Transform raw engine output to application output.

        Parameters
        ----------
        model_output : np.ndarray (N, 4 + 1 + NC)
            Where:
            - N is the number of detected bounding boxes,
            - 4 are the coordinates of the bounding box,
            - 1 is the confidence score of the bounding box,
            - NC is the number of classes.
        orig_shape : tuple
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
            model_output = self.yolov8_to_yolov5(model_output)

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

        bboxes = scale_boxes(bboxes, (self.img_size,) * 2, orig_shape)
        return bboxes, scores, classes

    def postprocess_offset_theta(
        self,
        offset: np.array,
        theta: np.array,
        orig_shape: tuple,
        offset_buffer: float = 0.15,
        curve_fit: bool = False,
    ):
        """
        Get line points and score from offset and theta logits.

        Parameters
        ----------
        offset : np.array
            Offset vector (classification logits)
        theta : np.array
            Theta vector (classification logits)
        orig_shape : tuple
            Original image shape (height, width)
        offset_buffer : float, optional
            Buffer for offset-theta model
        method : str, optional
            'argmax' or 'curve_fit'

        Returns
        -------
        (offset, theta): tuple
            offset = (offset, offset_score)
            theta = (theta, theta_score)
        """

        offset, theta = softmax(offset), softmax(theta)

        if curve_fit:
            offset_prob, offset, _ = gaussian_curve_fit(offset)
            theta_prob, theta, _ = gaussian_curve_fit(theta)
        else:
            offset_prob, offset = offset.max(), offset.argmax() / len(offset)
            theta_prob, theta = theta.max(), theta.argmax() / len(theta)

        points = offset_theta_to_points(offset, theta, offset_buffer=offset_buffer)
        points = np.array(points).flatten() * self.img_size
        points = scale_boxes(points, (self.img_size,) * 2, orig_shape)
        score = offset_prob * theta_prob  # pseudo-probability (because of softmax)

        return points, score, -1

    def yolov8_to_yolov5(self, model_output):
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

    def close(self):
        # Free CUDA memory
        for inp in self.inputs:
            inp.device.free()
        for out in self.outputs:
            out.device.free()

        self.ctx.pop()
        # self.ctx.detach()
