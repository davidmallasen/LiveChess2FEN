"""
Executes some tests for the complete digitization of a chessboard.
"""
import os

import numpy as np
import onnxruntime
import sklearn
from tensorflow.keras.applications.imagenet_utils import preprocess_input as \
        prein_squeezenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as \
        prein_mobilenet
from tensorflow.keras.applications.xception import preprocess_input as \
        prein_xception
from tensorflow.keras.models import load_model

try:
    import pycuda.driver as cuda
    # pycuda.autoinit causes pycuda to automatically manage CUDA context
    # creation and cleanup.
    import pycuda.autoinit
    import tensorrt as trt
except ImportError:
    cuda = None
    trt = None

from lc2fen.predict_board import load_image
from lc2fen.test_predict_board import predict_board, print_fen_comparison


# PRE_INPUT example:
#   from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as
#       prein_mobilenet
ACTIVATE_KERAS = False
MODEL_PATH_KERAS = "selected_models/Xception_last.h5"
IMG_SIZE_KERAS = 299
PRE_INPUT_KERAS = prein_xception

ACTIVATE_ONNX = False
MODEL_PATH_ONNX = "selected_models/MobileNetV2_0p5_all.onnx"
IMG_SIZE_ONNX = 224
PRE_INPUT_ONNX = prein_mobilenet

ACTIVATE_TRT = False
MODEL_PATH_TRT = "selected_models/SqueezeNet1p1.trt"
IMG_SIZE_TRT = 227
PRE_INPUT_TRT = prein_squeezenet


class __HostDeviceTuple:
    """A tuple of host and device. Clarifies code."""

    def __init__(self, _host, _device):
        self.host = _host
        self.device = _device


def __allocate_buffers(engine):
    """Allocates all buffers required for the specified engine."""
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        # Get binding (tensor/buffer) size
        size = trt.volume(
            engine.get_binding_shape(binding)) * engine.max_batch_size
        # Get binding (tensor/buffer) data type (numpy-equivalent)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate page-locked memory (i.e., pinned memory) buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Allocate linear piece of device memory
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(__HostDeviceTuple(host_mem, device_mem))
        else:
            outputs.append(__HostDeviceTuple(host_mem, device_mem))

    stream = cuda.Stream()
    return inputs, outputs, bindings, stream


def __infer(context, bindings, inputs, outputs, stream, batch_size=64):
    """
    Infer outputs on the IExecutionContext for the specified inputs.
    """
    # Transfer input data to the GPU
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference
    context.execute_async(batch_size=batch_size, bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    stream.synchronize()

    return [out.host for out in outputs]


def read_correct_fen():
    """Reads the correct fen for testing from boards.fen file."""
    fens = []

    with open(os.path.join("predictions", "boards.fen"), 'r') as fen_fd:
        lines = fen_fd.read().splitlines()
        for line in lines:
            line = line.split()
            if len(line) != 2:
                raise ValueError("All lines in fen file must have the format "
                                 "'fen orientation'")
            fens.append(line[0])
    return fens


def test_predict_board(obtain_predictions):
    """Tests board prediction."""
    fens = read_correct_fen()

    fen = predict_board(os.path.join("predictions", "test1.jpg"), "BL",
                        obtain_predictions)
    print_fen_comparison("test1.jpg", fen, fens[0])

    fen = predict_board(os.path.join("predictions", "test2.jpg"), "BL",
                        obtain_predictions)
    print_fen_comparison("test2.jpg", fen, fens[1])

    fen = predict_board(os.path.join("predictions", "test3.jpg"), "BL",
                        obtain_predictions)
    print_fen_comparison("test3.jpg", fen, fens[2])

    fen = predict_board(os.path.join("predictions", "test4.jpg"), "TL",
                        obtain_predictions)
    print_fen_comparison("test4.jpg", fen, fens[3])

    fen = predict_board(os.path.join("predictions", "test5.jpg"), "TR",
                        obtain_predictions)
    print_fen_comparison("test5.jpg", fen, fens[4])


def main_keras():
    """Executes Keras test board predictions."""
    print("Keras predictions")
    model = load_model(MODEL_PATH_KERAS)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, IMG_SIZE_KERAS, PRE_INPUT_KERAS)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    test_predict_board(obtain_pieces_probs)


def main_onnx():
    """Executes ONNXRuntime test board predictions."""
    print("ONNXRuntime predictions")
    sess = onnxruntime.InferenceSession(MODEL_PATH_ONNX)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, IMG_SIZE_ONNX, PRE_INPUT_ONNX)
            predictions.append(
                sess.run(None, {sess.get_inputs()[0].name: piece_img})[0][0])
        return predictions

    test_predict_board(obtain_pieces_probs)


def main_tensorrt():
    """Executes TensorRT test board predictions."""
    print("TensorRT predictions")
    if cuda is None or trt is None:
        raise ImportError("Unable to import pycuda or tensorrt")

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    # Read and deserialize the serialized ICudaEngine
    with open(MODEL_PATH_TRT, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    inputs, outputs, bindings, stream = __allocate_buffers(engine)

    img_array = np.zeros(
        (engine.max_batch_size, trt.volume((IMG_SIZE_TRT, IMG_SIZE_TRT, 3))))

    # Create an IExecutionContext (context for executing inference)
    with engine.create_execution_context() as context:

        def obtain_pieces_probs(pieces):
            # Assuming batch size == 64
            for i, piece in enumerate(pieces):
                img_array[i] = load_image(piece, IMG_SIZE_TRT,
                                          PRE_INPUT_TRT).ravel()
            np.copyto(inputs[0].host, img_array.ravel())
            trt_outputs = __infer(
                context, bindings, inputs, outputs, stream)[-1]

            return [trt_outputs[ind:ind + 13] for ind in range(0, 13 * 64, 13)]

        test_predict_board(obtain_pieces_probs)


if __name__ == "__main__":
    if ACTIVATE_KERAS:
        main_keras()
    if ACTIVATE_ONNX:
        main_onnx()
    if ACTIVATE_TRT:
        main_tensorrt()
