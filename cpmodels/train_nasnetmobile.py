"""
Train NASNetMobile model.
"""
from keras.applications import NASNetMobile
from keras.applications.nasnet import preprocess_input
from keras.engine.saving import load_model

from chess_piece_models_common import build_model, data_generators, \
    train_model, plot_model_history, evaluate_model, model_callbacks


def train_chesspiece_model():
    """Trains the chesspiece model based on NASNetMobile."""
    base_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False,
                              weights='imagenet')

    # First train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model = build_model(base_model)

    train_generator, validation_generator = data_generators(
        preprocess_input, (224, 224), 64)

    callbacks = model_callbacks(5, "./models/NASNetMobile_pre.h5", 0.1, 10)

    history = train_model(model, 20, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/NASNetMobile_pre_acc.png",
                       "./models/NASNetMobile_pre_loss.png")
    evaluate_model(model, validation_generator)

    # Also train 10-12
    for layer in model.layers[:635]:
        layer.trainable = False
    for layer in model.layers[635:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/NASNetMobile.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/NASNetMobile_acc.png",
                       "./models/NASNetMobile_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/NASNetMobile_last.h5")


def continue_training():
    """Continues training the chesspiece model based on NASNetMobile."""
    model = load_model("./models/NASNetMobile.h5")

    train_generator, validation_generator = data_generators(
        preprocess_input, (224, 224), 64)

    # Train all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/NASNetMobile_all.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/NASNetMobile_all_acc.png",
                       "./models/NASNetMobile_all_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/NASNetMobile_all_last.h5")


if __name__ == "__main__":
    train_chesspiece_model()
    continue_training()
