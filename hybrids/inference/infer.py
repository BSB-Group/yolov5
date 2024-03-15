"""
Abstract class for inference.
"""

from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np


class Infer(ABC):
    """
    Abstract class for inference.
    """

    @property
    @abstractmethod
    def input_shape(self) -> Union[tuple, List[tuple]]:
        """Return input(s) shape of the model."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Return data type of the model."""

    @property
    @abstractmethod
    def fp16(self) -> bool:
        """Return if model is using FP16 precision."""

    @property
    def input_hw(self) -> tuple:
        """Return input(s) height and width of the model."""
        shapes = self.input_shape
        if isinstance(shapes, list):
            return [shape[-2:] for shape in shapes]
        return shapes[-2:]

    @property
    def input_ch(self) -> int:
        """Return input(s) number of channels of the model."""
        shapes = self.input_shape
        if isinstance(shapes, list):
            return [shape[1] for shape in shapes]
        return shapes[1]

    @property
    def input_bs(self) -> int:
        """Return input(s) batch size of the model."""
        shapes = self.input_shape
        if isinstance(shapes, list):
            return [shape[0] for shape in shapes]
        return shapes[0]

    _alloc_arrays = None

    @property
    def alloc_arrays(self) -> List[np.ndarray]:
        """Return list of arrays to allocate memory for multiple usage."""
        if self._alloc_arrays is None:
            if isinstance(self.input_shape, list):
                self._alloc_arrays = [
                    np.zeros(shape, dtype=dtype)
                    for dtype, shape in zip(self.dtype, self.input_shape)
                ]
            else:
                self._alloc_arrays = np.zeros(self.input_shape, dtype=self.dtype)
        return self._alloc_arrays

    @abstractmethod
    def forward(self, ims) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference on batch of images.

        Parameters
        ----------
        ims : np.ndarray | list[np.ndarray]
            Batch of images to process.

        Returns
        -------
        np.ndarray | list[np.ndarray]
            Model outputs.
        """
