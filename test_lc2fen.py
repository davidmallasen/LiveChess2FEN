"""This script contains tests for chessboard digitization."""


# `sklearn` is required for Jetson (to avoid "cannot allocate memory in
# static TLS block" error)
import sklearn
from keras.applications.imagenet_utils import (
    preprocess_input as prein_squeezenet,
)
from keras.applications.mobilenet_v2 import preprocess_input as prein_mobilenet
from keras.applications.xception import preprocess_input as prein_xception

from lc2fen.predict_board import (
    predict_board_keras,
    predict_board_onnx,
    predict_board_trt,
)


ACTIVATE_KERAS = False
MODEL_PATH_KERAS = "data/models/Xception_last.h5"
IMG_SIZE_KERAS = 299
PRE_INPUT_KERAS = prein_xception

ACTIVATE_ONNX = False
MODEL_PATH_ONNX = "data/models/MobileNetV2_0p5_all.onnx"
IMG_SIZE_ONNX = 224
PRE_INPUT_ONNX = prein_mobilenet

ACTIVATE_TRT = False
MODEL_PATH_TRT = "data/models/SqueezeNet1p1.trt"
IMG_SIZE_TRT = 227
PRE_INPUT_TRT = prein_squeezenet


def main_keras():
    """Execute the Keras-based board-prediction tests."""
    print("Keras predictions")
    predict_board_keras(
        MODEL_PATH_KERAS, IMG_SIZE_KERAS, PRE_INPUT_KERAS, test=True
    )


def main_onnx():
    """Execute the ONNXRuntime-based board-prediction tests."""
    print("ONNXRuntime predictions")
    predict_board_onnx(
        MODEL_PATH_ONNX, IMG_SIZE_ONNX, PRE_INPUT_ONNX, test=True
    )


def main_tensorrt():
    """Execute the TensorRT-based board-prediction tests."""
    print("TensorRT predictions")
    predict_board_trt(MODEL_PATH_TRT, IMG_SIZE_TRT, PRE_INPUT_TRT, test=True)


if __name__ == "__main__":
    if ACTIVATE_KERAS:
        main_keras()
    if ACTIVATE_ONNX:
        main_onnx()
    if ACTIVATE_TRT:
        main_tensorrt()
