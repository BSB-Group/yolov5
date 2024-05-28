"""
Benchmarking script for the AHOY and DAN models.

Example usage:
    python benchmark.py --engine-path /path/to/ahoy.engine
    python benchmark.py --engine-path /path/to/dan.engine
"""

import argparse
from typing import Sequence, Tuple
from urllib.request import urlopen

import numpy as np
from inference.ahoy import AHOYv5
from inference.dan import DANv5
from PIL import Image


def get_dan_model_and_input(engine_path: str) -> Tuple[DANv5, Sequence[np.ndarray]]:
    """Get the DAN model and input images for testing."""
    dan = DANv5(model_path=engine_path)

    url_rgb = "https://github.com/SEA-AI/.github/blob/main/assets/example_1_RGB.jpg?raw=true"
    url_ir = "https://github.com/SEA-AI/.github/blob/main/assets/example_1_IR.jpg?raw=true"

    image_rgb = np.array(Image.open(urlopen(url_rgb)).convert("RGB"))
    image_ir = np.array(Image.open(urlopen(url_ir)).convert("RGB"))

    image_rgb = np.stack([image_rgb for _ in range(dan.model.input_bs[0])], axis=0)
    image_ir = np.stack([image_ir for _ in range(dan.model.input_bs[1])], axis=0)
    print(image_rgb.shape, image_ir.shape)

    return dan, (image_rgb, image_ir)


def get_ahoy_model_and_input(engine_path: str) -> Tuple[AHOYv5, np.ndarray]:
    """Get the AHOY model and input image for testing."""
    ahoy = AHOYv5(model_path=engine_path)
    if 640 in ahoy.model.input_hw:
        url = "https://github.com/SEA-AI/.github/blob/main/assets/example_1_IR.jpg?raw=true"
    else:
        url = "https://github.com/SEA-AI/.github/blob/main/assets/example_1_RGB.jpg?raw=true"

    image = np.array(Image.open(urlopen(url)).convert("RGB"))
    image = np.stack([image for _ in range(ahoy.model.input_bs)], axis=0)
    return ahoy, image


def main(engine_path: str, n: int = 100, dry_run: bool = False):
    """Basic benchmarking of the AHOY and DAN models."""

    if "dan" in engine_path:
        model, images = get_dan_model_and_input(engine_path)
    elif "ahoy" in engine_path:
        model, images = get_ahoy_model_and_input(engine_path)
    else:
        raise ValueError("Cannot determine model type from engine path.")

    if not dry_run:
        print(f"benchmarking over {n} iterations...")
        for _ in range(n):
            _ = model(images)

        # print average times
        for name, profile in model.profiles.items():
            print(f"{profile.t / n * 1E3:>5.1f} ms [avg] - {name}")

    model.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Test DANv5")
    parser.add_argument(
        "-e",
        "--engine-path",
        type=str,
        required=True,
        help="Path to the TensorRT engine file",
    )
    parser.add_argument("-n", type=int, default=100, help="Number of iterations to benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(**vars(args))
