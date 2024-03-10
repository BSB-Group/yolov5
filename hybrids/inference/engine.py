"""
Utility functions for building/saving/loading TensorRT Engine
Taken mostly from this repo
https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/common.py
"""

import json
import tensorrt as trt
import pycuda.driver as cuda


class HostDeviceMem(object):
    # Simple helper data class that's a little nicer to use than a 2-tuple.

    def __init__(self, host_mem, device_mem, shape, name):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.dtype = host_mem.dtype
        self.name = name

    def __str__(self):
        return (
            str(self.host)
            + "\nDevice:\n * "
            + str(self.device)
            + "\nShape:\n * "
            + str(self.shape)
            + "\nDtype:\n * "
            + str(self.dtype)
            + "\nName:\n * "
            + str(self.name)
        )

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, shape, binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, shape, binding))
    return inputs, outputs, bindings, stream


def save_engine(engine, engine_dest_path):
    """Saves a serialized engine to a file."""
    print("Engine:", engine)
    buf = engine.serialize()
    with open(engine_dest_path, "wb") as f:
        f.write(buf)


def load_engine(engine_path, logger=trt.Logger):
    """Loads a serialized engine from a file."""
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        if "yolov8" in engine_path:
            meta_len = int.from_bytes(
                f.read(4), byteorder="little"
            )  # read metadata length
            metadata = json.loads(f.read(meta_len).decode("utf-8"))
            print("metadata:", metadata)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def do_inference(context, bindings, inputs, outputs, stream):
    """
    This function is generalized for multiple inputs/outputs.
    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
