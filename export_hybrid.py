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
        --engine-fname ahoy.engine

    python export_hybrid.py \
        --det-weights /path/to/obj_det_weights.pt /path/to/ir_obj_det_weights.pt \
        --hor-weights /path/to/hor_det_weights.pt /path/to/ir_hor_det_weights.pt \
        --imgsz 1280 640 \
        --batch-size 2 2 \
        --half \
        --fuse \
        --engine-fname dan.engine
"""

from typing import List, Union, Sequence
import argparse
from pathlib import Path
import torch

from models.custom import AHOY, DAN
from models.yolo import Detect
from export import export_engine


def init_model(
    det_weights: List[str],
    hor_weights: List[str],
    half: bool,
    fuse: bool,
) -> torch.nn.Module:
    """
    Load the AHOY or DAN model based on the number of weights provided.
    """
    if len(det_weights) == len(hor_weights) == 1:
        return AHOY(
            obj_det_weigths=det_weights[0],
            hor_det_weights=hor_weights[0],
            fp16=half,
            fuse=fuse,
        )
    if len(det_weights) == len(hor_weights) == 2:
        return DAN(
            rgb_model=AHOY(
                obj_det_weigths=det_weights[0],
                hor_det_weights=hor_weights[0],
                fp16=half,
                fuse=fuse,
            ),
            ir_model=AHOY(
                obj_det_weigths=det_weights[1],
                hor_det_weights=hor_weights[1],
                fp16=half,
                fuse=fuse,
            ),
        )
    raise ValueError("Invalid number of weights.")


def get_dummy_input(
    batch_size: int, imgsz: int, half: bool, device: str
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """
    Create a dummy input image.
    """
    dummy_input = [
        torch.zeros((bs, 3, sz, sz), device=device) for bs, sz in zip(batch_size, imgsz)
    ]
    if half:
        dummy_input = [inp.half() for inp in dummy_input]
    if len(dummy_input) == 1:
        return dummy_input[0]
    return dummy_input


def main(
    det_weights: List[str],
    hor_weights: List[str],
    imgsz: List[int],
    batch_size: List[int],
    half: bool,
    fuse: bool,
    engine_fname: str = None,
):
    """
    Export the model to TensorRT engine.
    """

    model = init_model(det_weights, hor_weights, half, fuse)
    print(f"🚀 Exporting model {type(model).__name__} to {engine_fname}...")

    if not engine_fname:
        engine_fname = f"{type(model).__name__.lower()}_b{batch_size}.engine"

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

    # Create dummy input
    image = get_dummy_input(batch_size, imgsz, half, model.device)
    # need to run once to get the model to JIT compile
    if isinstance(image, (list, tuple)):
        print(f"🔮 Dummy input...{[im.shape for im in image]}")
        model(*image)
    else:
        print(f"🔮 Dummy input...{image.shape}")
        model(image)

    export_engine(
        model,
        im=image,
        file=Path(engine_fname),
        half=half,
        dynamic=dynamic,
        simplify=False,
    )
    print(f"🎉 Model successfully exported to {engine_fname}! Ready to deploy! 🚀")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", nargs='+', type=str, required=True,
                        help="List of paths to the detection model weights.")
    parser.add_argument("--hor-weights", nargs='+', type=str, required=True,
                        help="List of paths to the horizontal model weights.")
    parser.add_argument("--imgsz", nargs='+', type=int, default=[640],
                        help="Image sizes (width and height can be different).")
    parser.add_argument("--batch-size", nargs='+', type=int, default=[1],
                        help="Batch sizes for processing.")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision (FP16) for inference.")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse convolution and batchnorm layers.")
    parser.add_argument("--engine-fname", type=str, default="model.engine",
                        help="Filename for the exported TensorRT engine.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(**vars(args))
