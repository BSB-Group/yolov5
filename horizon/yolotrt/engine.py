# Utility functions for building/saving/loading TensorRT Engine
import json
import tensorrt as trt
import pycuda.driver as cuda


class HostDeviceMem(object):
    # Simple helper data class that's a little nicer to use than a 2-tuple.

    def __init__(self, host_mem, device_mem, shape):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.dtype = host_mem.dtype

    def __str__(self):
        return str(self.host) \
            + "\nDevice:\n * " + str(self.device) \
            + "\nShape:\n * " + str(self.shape) \
            + "\nDtype:\n * " + str(self.dtype)

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
            inputs.append(HostDeviceMem(host_mem, device_mem, shape))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, shape))
    return inputs, outputs, bindings, stream


def save_engine(engine, engine_dest_path):
    print('Engine:', engine)
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)


def load_engine(engine_path, logger=trt.Logger):
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        if 'yolov8' in engine_path:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
            print('metadata:', metadata)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

