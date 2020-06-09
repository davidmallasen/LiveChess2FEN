import argparse
import os

from keras.applications.imagenet_utils import preprocess_input as \
    prein_squeezenet
from keras.applications.mobilenet_v2 import preprocess_input as prein_mobilenet

from lc2fen.predict_board import predict_board_keras, predict_board_onnx, \
    predict_board_trt

ACTIVATE_KERAS = False
MODEL_PATH_KERAS = "selected_models/SqueezeNet1p1.h5"
IMG_SIZE_KERAS = 227
PRE_INPUT_KERAS = prein_squeezenet

ACTIVATE_ONNX = False
MODEL_PATH_ONNX = "selected_models/MobileNetV2_0p5_all.onnx"
IMG_SIZE_ONNX = 224
PRE_INPUT_ONNX = prein_mobilenet

ACTIVATE_TRT = False
MODEL_PATH_TRT = "selected_models/MobileNetV2_0p5_all.trt"
IMG_SIZE_TRT = 224
PRE_INPUT_TRT = prein_mobilenet


def parse_arguments():
    """Parses the script arguments and sets the corresponding flags.
    Returns the path of the image or folder and the location of the a1
    square."""
    global ACTIVATE_KERAS, ACTIVATE_ONNX, ACTIVATE_TRT

    parser = argparse.ArgumentParser(
        description="Predicts board configurations from images.")

    parser.add_argument("path",
                        help="Path to the image or folder you wish to predict")
    parser.add_argument("a1_pos",
                        help="Location of the a1 square in the chessboard "
                             "(B = bottom, T = top, R = right, L = left)",
                        choices=["BL", "BR", "TL", "TR"])

    inf_engine = parser.add_mutually_exclusive_group(required=True)
    inf_engine.add_argument("-k", "--keras",
                            help="Run inference using Keras",
                            action="store_true")
    inf_engine.add_argument("-o", "--onnx",
                            help="Run inference using ONNXRuntime",
                            action="store_true")
    inf_engine.add_argument("-t", "--trt",
                            help="Run inference using TensorRT",
                            action="store_true")

    args = parser.parse_args()

    if args.keras:
        ACTIVATE_KERAS = True
    elif args.onnx:
        ACTIVATE_ONNX = True
    elif args.trt:
        ACTIVATE_TRT = True
    else:
        ValueError("No inference engine selected. This should be unreachable.")

    return args.path, args.a1_pos


def main():
    """Parses the arguments and prints the predicted FEN."""
    path, a1_pos = parse_arguments()
    is_dir = os.path.isdir(path)
    if ACTIVATE_KERAS:
        fen = predict_board_keras(MODEL_PATH_KERAS, IMG_SIZE_KERAS,
                                  PRE_INPUT_KERAS, path, a1_pos, is_dir)
    elif ACTIVATE_ONNX:
        fen = predict_board_onnx(MODEL_PATH_ONNX, IMG_SIZE_ONNX,
                                 PRE_INPUT_ONNX, path, a1_pos, is_dir)
    elif ACTIVATE_TRT:
        fen = predict_board_trt(MODEL_PATH_TRT, IMG_SIZE_TRT,
                                PRE_INPUT_TRT, path, a1_pos, is_dir)
    else:
        fen = None
        ValueError("No inference engine selected. This should be unreachable.")

    print(fen)


if __name__ == "__main__":
    main()
