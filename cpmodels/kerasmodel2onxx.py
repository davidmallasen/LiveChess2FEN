"""
Convert Keras model to ONNX format.
"""
import tensorflow as tf
import tf2onnx

MODELS_PATH = "../selected_models/"
KERAS_MODEL_NAME = "MobileNetV2_0p5_all.h5"
ONNX_MODEL_NAME = KERAS_MODEL_NAME[:-2] + "onnx"

image_size = 224
channels = 3
batch_size = 1
target_opset = 13


def main():
    # Convert the .h5 keras model to .pb
    keras_model = tf.keras.models.load_model(MODELS_PATH + KERAS_MODEL_NAME)
    keras_model.save(MODELS_PATH + "saved_model")

    # Convert to ONNX
    spec = (tf.TensorSpec((batch_size, image_size, image_size, channels), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(keras_model, 
                                                input_signature=spec, 
                                                opset=target_opset, 
                                                output_path=MODELS_PATH + ONNX_MODEL_NAME)


if __name__ == "__main__":
    main()
