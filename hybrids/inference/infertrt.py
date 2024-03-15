"""
Inference using TensorRT engine.
"""

from typing import Union, List
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from .infer import Infer
from .engine import load_engine, allocate_buffers, do_inference

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class InferTRT(Infer):
    """
    Inference using TensorRT engine.
    """

    def __init__(self, trt_engine_path: str, device: int = 0) -> None:
        """
        Load TensorRT engine and allocate memory for inference.

        Parameters
        ----------
        trt_engine_path : str
            Path to the TensorRT engine file.
        device : int, optional
            CUDA device to use.
        """

        cuda.init()  # needed otherwise: cuDeviceGet failed: initialization error
        self.ctx = cuda.Device(device).make_context()

        # Display requested engine settings to stdout
        print(f"Loading TensorRT inference engine {trt_engine_path}")
        self.trt_engine = load_engine(trt_engine_path, TRT_LOGGER)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.trt_engine
        )

        for i, inp in enumerate(self.inputs):
            print(f" - Input {i}: shape {inp.shape}, dtype {inp.dtype}")

    @property
    def input_shape(self) -> tuple:
        """Return input(s) shape of the model."""
        shapes = [inp.shape for inp in self.inputs]
        if len(shapes) == 1:
            return shapes[0]
        return shapes

    @property
    def dtype(self) -> np.dtype:
        """Return data type of the model."""
        dtypes = [inp.dtype for inp in self.inputs]
        if len(dtypes) == 1:
            return dtypes[0]
        return dtypes

    @property
    def fp16(self) -> bool:
        """Return if model is using FP16 precision."""
        return np.all([inp.dtype == np.float16 for inp in self.inputs])

    def check_input_shape(self, ims: Union[np.ndarray, List[np.ndarray]]) -> None:
        """
        Check if input shape matches image shape.

        Parameters
        ----------
        ims : np.ndarray | list[np.ndarray]
            Image to process.
        """

        # Check if input shape matches image shape
        if len(self.inputs) == 1:
            ims = [ims]

        for inp, im in zip(self.inputs, ims):
            msg = f"input shape {inp.shape} not equal to image shape {im.shape}"
            assert inp.shape == im.shape, msg

    def forward(
        self, ims: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
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

        # Check if input shape matches image shape
        self.check_input_shape(ims)

        # NOTE: when using multiple threads, make sure to handle CUDA context
        # https://forums.developer.nvidia.com/t/how-to-use-tensorrt-by-the-multi-threading-package-of-python/123085/8
        # threading.Thread.__init__(self)
        self.ctx.push()

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        for inp, im in zip(self.inputs, [ims] if len(self.inputs) == 1 else ims):
            np.copyto(inp.host, np.ascontiguousarray(im.ravel()))

        # When infering on single image, we measure inference
        # time to output it to the user
        # inference_start_time = time.time()

        # Fetch output from the model
        detection_outs = do_inference(
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

    def close(self):
        """Free CUDA memory and detach context."""

        for inp in self.inputs:
            inp.device.free()
        for out in self.outputs:
            out.device.free()

        self.ctx.pop()
        # self.ctx.detach()
