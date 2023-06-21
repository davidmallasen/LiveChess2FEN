"""
Train SqueezeNet-v1.1 model.
"""
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.saving import load_model

from squeezenet import SqueezeNet
from chess_piece_models_common import build_model, data_generators, \
    train_model, plot_model_history, evaluate_model, model_callbacks


def train_chesspiece_model():
    """Trains the chesspiece model based on SqueezeNet-v1.1."""
    base_model = SqueezeNet(input_shape=(227, 227, 3), include_top=False,
                            weights='imagenet')

    # First train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model = build_model(base_model)

    train_generator, validation_generator = data_generators(
        preprocess_input, (227, 227), 64)

    callbacks = model_callbacks(5, "./models/SqueezeNet1p1_pre.h5", 0.1, 10)

    history = train_model(model, 20, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/SqueezeNet1p1_pre_acc.png",
                       "./models/SqueezeNet1p1_pre_loss.png")
    evaluate_model(model, validation_generator)

    # Also train fire 7-9
    for layer in model.layers[:41]:
        layer.trainable = False
    for layer in model.layers[41:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/SqueezeNet1p1.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/SqueezeNet1p1_acc.png",
                       "./models/SqueezeNet1p1_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/SqueezeNet1p1_last.h5")


def continue_training():
    """Continues training the chesspiece model based on SqueezeNet-v1.1.
    """
    model = load_model("./models/SqueezeNet1p1.h5")

    train_generator, validation_generator = data_generators(
        preprocess_input, (227, 227), 64)

    # Train all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/SqueezeNet1p1_all.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/SqueezeNet1p1_all_acc.png",
                       "./models/SqueezeNet1p1_all_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/SqueezeNet1p1_all_last.h5")


if __name__ == "__main__":
    train_chesspiece_model()
    continue_training()
