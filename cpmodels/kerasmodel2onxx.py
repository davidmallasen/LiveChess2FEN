"""
Convert Keras model to ONNX format.

After converting the model to onnx format, to create the TensorRT
engine using onnx2trt execute the following command:

onnx2trt MobileNetV2_2.onnx -o MobileNetV2_2_engine.trt -b 1
  -w 268435456 -d 16 -lgv
"""
import keras2onnx

from keras.applications import NASNetMobile, MobileNetV2
from keras.models import load_model, clone_model

from chess_piece_models_common import build_model
from squeezenet import SqueezeNet
from train_alexnet import alexnet

KERAS_MODEL_PATH = "../selected_models/NASNetMobile_all_last.h5"
ONNX_MODEL_PATH = KERAS_MODEL_PATH[:-2] + "onnx"

image_size = 224
channels = 3
batch_size = 1
target_opset = 8


def main():
    """Converts a keras model into ONNX format."""
    # model = alexnet((224, 224, 3))
    model = build_model(NASNetMobile(input_shape=(224, 224, 3),
                                     include_top=False,
                                     weights='imagenet'))
    model.load_weights(KERAS_MODEL_PATH)

    # If we have not specified explicitly image dimensions when creating
    # the model
    #
    # model = load_model(KERAS_MODEL_PATH)
    # model._layers[0].batch_input_shape = (batch_size, image_size, image_size,
    #                                       channels)
    #
    # In order for the input_shape to be saved correctly we have to
    # clone the model into a new one
    #
    # model = clone_model(model)
    #
    # When cloning we loose the weights, load them again
    #
    # model.load_weights(KERAS_MODEL_PATH)

    onnx_model = keras2onnx.convert_keras(model, model.name)

    # target_opset=target_opset,
    # debug_mode=True

    keras2onnx.save_model(onnx_model, ONNX_MODEL_PATH)


if __name__ == "__main__":
    main()
