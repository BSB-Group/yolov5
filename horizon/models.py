from pathlib import Path
import torch
from torch import nn

from utils.plots import feature_visualization
from utils.general import LOGGER, non_max_suppression
from utils.torch_utils import select_device
from models.experimental import attempt_load
from models.yolo import BaseModel, DetectionModel
from models.common import Classify, DetectMultiBackend


class HorizonModel(BaseModel):
    """
    YOLOv5 backbone + classification heads for pitch and theta.
    """

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        nc_pitch: int = 500,
        nc_theta: int = 500,
        device: str | torch.device = None,  # automatically select device
        cutoff: int = None,
        fp16: bool = False,
        fuse: bool = False,  # false for training, true for inference
    ):
        """
        Horizon detection model.

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            device (str, optional): device to run model on. Defaults to ''.
            cutoff (int, optional): cutoff layer for classification heads.
                If not specified, the SPPF layer is used as cutoff. Defaults to None.
        """
        super().__init__()

        assert weights is not None, "weights must be specified"
        assert isinstance(nc_pitch, int), "nc_pitch must be an integer"
        assert isinstance(nc_theta, int), "nc_theta must be an integer"
        assert isinstance(fp16, bool), "fp16 must be a boolean"

        self.nc_pitch = nc_pitch
        self.nc_theta = nc_theta
        self.device = select_device(device)
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=fuse)
            stride = model.stride
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        if isinstance(model, DetectionModel):
            LOGGER.warning(
                "WARNING ⚠️ converting YOLOv5 DetectionModel to HorizonModel"
            )
            self.cutoff = _find_cutoff(model) if cutoff is None else cutoff
            self._from_detection_model(model, self.cutoff)  # inplace modification

        self.model = model.model
        self.model.to(self.device)
        self.model.half() if fp16 else self.model.float()
        self.stride = stride
        self.save = model.save

    def _from_detection_model(self, model: DetectionModel, cutoff: int):
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend

        c_pitch, c_theta = _get_classification_heads(
            model, cutoff, self.nc_pitch, self.nc_theta
        )
        model.save = set(list(model.save + [cutoff]))  # add cutoff to save

        # remove layers after cutoff
        model.model = model.model[: self.cutoff + 1]

        # add classification heads to model
        model.model.add_module(c_pitch.i, c_pitch)
        model.model.add_module(c_theta.i, c_theta)

    def _forward_once(self, x, profile=False, visualize=False):
        x_pitch, x_theta = None, None
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x)
            if m.type == "models.common.Classify" and "pitch" in m.i:
                x_pitch = m(x)
            elif m.type == "models.common.Classify" and "theta" in m.i:
                x_theta = m(x)
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x_pitch, x_theta)


class ObjectsModel(BaseModel):
    """
    Wrapper around YOLOv5 DetectionModel.
    """

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        device: str | torch.device = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = False,
    ):
        super().__init__()
        self.device = select_device(device)
        self.fp16 = fp16

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=fuse)
            stride = model.stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        self.model = model.model
        self.model.to(self.device)
        self.model.half() if fp16 else self.model.float()
        self.stride = stride
        self.names = names
        self.save = model.save


class AHOY(nn.Module):
    """
    A
    H-orizon
    O-bject detection
    Y-OLOv5
    """

    # Ensemble of models
    def __init__(
        self,
        obj_det_weigths: str,
        hor_det_weights: str,
        device: str | torch.device = None,  # automatically select device
        fp16: bool = False,
        fuse: bool = True, # fuse conv and bn layers
        inplace: bool = True, # inplace modification of models
    ):
        super().__init__()
        self.obj_det = ObjectsModel(
            obj_det_weigths, device=device, fp16=fp16, fuse=fuse
        )
        self.hor_det = HorizonModel(
            hor_det_weights, device=device, fp16=fp16, fuse=fuse
        )
        self.device = select_device(device)
        self.fp16 = fp16
        self.stride = self.obj_det.stride
        self.names = self.obj_det.names

    def forward(self, x, profile=False, visualize=False):
        """
        Forward pass through models.
        """
        objects = self.obj_det(x, profile, visualize)
        pitch, theta = self.hor_det(x, profile, visualize)
        return objects, pitch, theta


class Hydra(BaseModel):
    """
    Model with two heads: object detection and horizon detection.
    HydraModel is a wrapper around YOLOv5 DetectionModel.
    In Greek mythology, Hydra is a serpent-like monster with many heads.
    """

    def __init__(
        self,
        weights: str = "yolov5n.pt",
        nc_pitch: int = 500,
        nc_theta: int = 500,
        device: str | torch.device = None,  # automatically select device
        cutoff: int = None,
        fp16: bool = False,
        task: str = "both",  # "detection", "horizon", "both"
    ):
        """
        NOTE: Not tested!!!
        Multi-task model with object detection and horizon detection.
        backbone --> neck --> detection heads
                 └-> classification heads for pitch and theta

        Args:
            model (DetectionModel): YOLOv5 model
            nc_pitch (int, optional): number of classes for pitch classification. Defaults to 500.
            nc_theta (int, optional): number of classes for theta classification. Defaults to 500.
            device (str, optional): device to run model on. Defaults to ''.
            cutoff (int, optional): cutoff layer for classification heads.
                If not specified, the SPPF layer is used as cutoff. Defaults to None.
            task (str, optional): task to run. Defaults to "both".
                Possible values: "detection", "horizon", "both"
        """
        super().__init__()

        assert weights is not None, "weights must be specified"
        assert isinstance(nc_pitch, int), "nc_pitch must be an integer"
        assert isinstance(nc_theta, int), "nc_theta must be an integer"
        assert isinstance(fp16, bool), "fp16 must be a boolean"
        assert task in [
            "detection",
            "horizon",
            "both",
        ], "task must be one of 'detection', 'horizon', 'both'"

        self.nc_pitch = nc_pitch
        self.nc_theta = nc_theta
        self.device = select_device(device)
        self.fp16 = fp16
        self.task = task

        if Path(weights).is_file() or weights.endswith(".pt"):
            model = attempt_load(weights, device="cpu", fuse=False)
            LOGGER.info(f"Loaded weights from {weights}")
        else:
            raise ValueError("model must be a path to a .pt file")

        if isinstance(model, DetectionModel):
            LOGGER.warning(
                "WARNING ⚠️ converting YOLOv5 DetectionModel to HorizonModel"
            )
            self.cutoff = _find_cutoff(model) if cutoff is None else cutoff
            self._add_classification_heads(model, self.cutoff)  # inplace modification

        self.model = model.model
        self.model.to(self.device)
        self.model.half() if fp16 else self.model.float()
        self.save = model.save
        self.stride = model.stride
        self.nc = model.nc

    def _add_classification_heads(self, model: DetectionModel, cutoff: int):
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend

        c_pitch, c_theta = _get_classification_heads(
            model, cutoff, self.nc_pitch, self.nc_theta
        )
        model.save = set(list(model.save + [cutoff]))  # add cutoff to save

        # add classification heads to model
        model.model.add_module(c_pitch.i, c_pitch)
        model.model.add_module(c_theta.i, c_theta)

    def _horizon_once(self, x, profile=False, visualize=False):
        x_pitch, x_theta = None, None
        y, dt = [], []  # outputs
        for m in self.model:
            if isinstance(m.i, int) and m.i > self.cutoff:
                continue
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.type == "models.common.Classify" and "pitch" in m.i:
                x_pitch = m(x)
            elif m.type == "models.common.Classify" and "theta" in m.i:
                x_theta = m(x)
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x_pitch, x_theta)

    def _detect_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.type == "models.common.Classify":
                continue
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _forward_once(self, x, profile=False, visualize=False):
        x_pitch, x_theta = None, None
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.type == "models.common.Classify" and "pitch" in m.i:
                x_pitch = m(x)
            elif m.type == "models.common.Classify" and "theta" in m.i:
                x_theta = m(x)
            else:  # object detection flow
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)

        return (x_pitch, x_theta)

        return (x, x_pitch, x_theta)

    def forward(self, x):
        if self.task == "detection":
            return self._detect_once(x)
        elif self.task == "horizon":
            return self._horizon_once(x)
        else:
            return self._forward_once(x)


def _find_cutoff(model):
    """
    Find cutoff layer for classification heads.
    """
    for i, m in enumerate(model.model):
        if m.type == "models.common.SPPF":
            return i - 1
    raise ValueError("Could not find cutoff layer for classification heads.")


def _get_classification_heads(model, cutoff, nc_pitch, nc_theta):
    """
    Get classification heads. Similar to method found in
    `models.yolo.ClassificationModel._from_detection_model`
    """

    # get number of input channels for classification heads
    m = model.model[cutoff + 1]  # layer after cutoff
    ch = (
        m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels
    )  # ch into module

    # define classification heads
    c_pitch = Classify(ch, nc_pitch)
    c_pitch.i, c_pitch.f, c_pitch.type = "c_pitch", cutoff, "models.common.Classify"
    c_theta = Classify(ch, nc_theta)
    c_theta.i, c_theta.f, c_theta.type = "c_theta", cutoff, "models.common.Classify"

    return c_pitch, c_theta
