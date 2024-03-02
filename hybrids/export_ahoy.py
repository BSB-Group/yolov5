import argparse
from pathlib import Path
import torch

from models.custom import AHOY
from models.yolo import Detect
from export import export_engine


def main(
    det_weights,
    hor_weights,
    imgsz,
    batch_size,
    half,
    fuse,
):
    ahoy = AHOY(
        obj_det_weigths=det_weights,
        hor_det_weights=hor_weights,
        fp16=half,
        fuse=fuse,
    )

    inplace = False  # default
    dynamic = False  # default

    # Update model
    ahoy.eval()
    for k, m in ahoy.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    image = torch.zeros((batch_size, 3, imgsz, imgsz), device=ahoy.device)
    # need to run once to get the model to JIT compile
    ahoy(image.half() if ahoy.fp16 else image)

    export_engine(
        ahoy,
        im=image.half() if half else image,
        file=Path("ahoy.engine"),
        half=ahoy.fp16,
        # opset=12,
        dynamic=dynamic,
        simplify=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", type=str)
    parser.add_argument("--hor-weights", type=str)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
