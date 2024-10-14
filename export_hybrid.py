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

    python export_hybrid.py \
        --det-weights /path/to/obj_det_weights.pt /path/to/ir_obj_det_weights.pt \
        --hor-weights /path/to/hor_det_weights.pt /path/to/ir_hor_det_weights.pt \
        --imgsz 1280 640 \
        --batch-size 2 2 \
        --half \
        --fuse \
        --fname dan.engine

For ONNX only, simple specify --fname model_name.onnx
"""

import argparse
from pathlib import Path
from typing import List, Sequence, Union

import torch

from export import export_engine
from models.custom import AHOY, DAN
from models.yolo import Detect


def init_model(
    det_weights: List[str],
    hor_weights: List[str],
    half: bool,
    fuse: bool,
) -> Union[AHOY, DAN]:
    """Load the AHOY or DAN model based on the number of weights provided."""
    if len(det_weights) == len(hor_weights) == 1:
        return AHOY(
            obj_det_weigths=det_weights[0],
            hor_det_weights=hor_weights[0],
            fp16=half,
            fuse=fuse,
        )
    if len(det_weights) == len(hor_weights) == 2:
        return DAN(
            model_a=AHOY(
                obj_det_weigths=det_weights[0],
                hor_det_weights=hor_weights[0],
                fp16=half,
                fuse=fuse,
            ),
            model_b=AHOY(
                obj_det_weigths=det_weights[1],
                hor_det_weights=hor_weights[1],
                fp16=half,
                fuse=fuse,
            ),
        )
    raise ValueError("Invalid number of weights.")


def get_dummy_input(
    batch_size: int, imgsz: int, device: str, fp32=False
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """Create a dummy input image."""
    dummy_input = [torch.zeros((bs, 3, sz, sz), device=device).byte() for bs, sz in zip(batch_size, imgsz)]
    if fp32:
        dummy_input = [inp.float() for inp in dummy_input]
    if len(dummy_input) == 1:
        return dummy_input[0]
    return dummy_input


def get_engine_fname(model, batch_size, imgsz):
    format_value = lambda x: "-".join(map(str, x)) if isinstance(x, list) else str(x)
    bs_str = format_value(batch_size)
    sz_str = format_value(imgsz)
    return f"{type(model).__name__.lower()}_b{bs_str}_sz{sz_str}.engine"


def main(
    det_weights: List[str],
    hor_weights: List[str],
    imgsz: List[int],
    batch_size: List[int],
    half: bool,
    fuse: bool,
    input_as_fp32: bool = False,
    fname: str = "",
):
    """Export the model to TensorRT engine."""

    model = init_model(det_weights, hor_weights, half, fuse)
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
    image = get_dummy_input(batch_size, imgsz, model.device, input_as_fp32)
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
    )
    print(f"🎉 Model successfully exported to {f}! Ready to deploy! 🚀")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dw",
        "--det-weights",
        nargs="+",
        type=str,
        required=True,
        help="List of paths to the detection model weights.",
    )
    parser.add_argument(
        "-hw",
        "--hor-weights",
        nargs="+",
        type=str,
        required=True,
        help="List of paths to the horizontal model weights.",
    )
    parser.add_argument(
        "-sz",
        "--imgsz",
        nargs="+",
        type=int,
        default=[640],
        help="Image sizes (width and height can be different).",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        nargs="+",
        type=int,
        default=[1],
        help="Batch sizes for processing.",
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
        "-ifp32",
        "--input-as-fp32",
        action="store_true",
        help="Input image as FP32.",
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
