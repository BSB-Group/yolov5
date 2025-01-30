"""
Export AHOY or DAN model to TensorRT engine.
Example usage:
    python export_hybrid.py \
        --det-weights /path/to/obj_det_weights.pt \
        --hor-weights /path/to/hor_det_weights.pt \
        --imgsz 640 \
        --batch-size 2 \
        --half \
        --fuse \
        --fname ahoy.engine

For ONNX only, simple specify --fname model_name.onnx
"""

import argparse
from pathlib import Path
from typing import Sequence, Union

import torch

from export import export_engine
from models.custom import AHOY
from models.yolo import Detect


def get_dummy_input(
    batch_size: int, imgsz: int, device: str, fp32=False
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """Create a dummy input image."""
    dummy_input = torch.zeros((batch_size, 3, imgsz, imgsz), device=device).byte()
    return dummy_input.float() if fp32 else dummy_input


def get_engine_fname(model, batch_size, imgsz):
    format_value = lambda x: "-".join(map(str, x)) if isinstance(x, list) else str(x)
    bs_str = format_value(batch_size)
    sz_str = format_value(imgsz)
    return f"{type(model).__name__.lower()}_b{bs_str}_sz{sz_str}.engine"


def main(
    det_weights: str,
    hor_weights: str,
    imgsz: int,
    batch_size: int,
    half: bool,
    fuse: bool,
    trt7_compatible: bool = False,
    fname: str = "",
):
    """Export the model to TensorRT engine."""

    model = AHOY(
        obj_det_weigths=det_weights,
        hor_det_weights=hor_weights,
        fp16=half,
        fuse=fuse,
    )

    if not fname:
        fname = get_engine_fname(model, batch_size, imgsz)
    print(f"🚀 Exporting model {type(model).__name__} to {fname}...")

    inplace = False  # default
    dynamic = False  # default

    # Update model
    model.eval()
    print("✨ Preparing the model for export...")
    for _, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
    model.register_io_hooks()  # inp: uint8 -> fp32/fp16 / 255.0, out: fp16 -> fp32

    # Create dummy input
    image = get_dummy_input(batch_size, imgsz, model.device, trt7_compatible)
    # need to run once to get the model to JIT compile
    if isinstance(image, (list, tuple)):
        print(f"🔮 Dummy input...{[(im.shape, im.dtype) for im in image]}")
        model(*image)
    else:
        print(f"🔮 Dummy input...{image.shape}")
        model(image)

    f, _ = export_engine(
        model,
        im=image,
        file=Path(fname),
        half=half,
        dynamic=dynamic,
        simplify=False,
        onnx_only=fname.endswith(".onnx"),
        trt_7_compatible=trt7_compatible,
    )
    print(f"🎉 Model successfully exported to {f}! Ready to deploy! 🚀")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dw",
        "--det-weights",
        type=str,
        required=True,
        help="Path to the detection model weights.",
    )
    parser.add_argument(
        "-hw",
        "--hor-weights",
        type=str,
        required=True,
        help="Path to the horizontal model weights.",
    )
    parser.add_argument(
        "-sz",
        "--imgsz",
        type=int,
        default=640,
        help="Image size (square).",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "-hp",
        "--half",
        action="store_true",
        help="Use half precision (FP16) for inference.",
    )
    parser.add_argument(
        "-fu",
        "--fuse",
        action="store_true",
        help="Fuse convolution and batchnorm layers.",
    )
    parser.add_argument(
        "-trt7",
        "--trt7-compatible",
        action="store_true",
        help="Export TensorRT 7 compatible model.",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default="",
        help="Filename for the exported model (engine or onnx).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(**vars(args))
