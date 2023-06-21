"""
Train AlexNet model.
"""
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from chess_piece_models_common import data_generators, train_model, \
    plot_model_history, evaluate_model, model_callbacks


def alexnet(input_shape=(224, 224, 3)):
    """AlexNet model."""
    model = Sequential()
    model.add(
        Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11),
               strides=(4, 4), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same",
               activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))
    model.add(
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
               activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=13, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def preprocess_input(x):
    """AlexNet preprocessing function."""
    x /= 127.5
    x -= 1.
    return x


def train_chesspiece_model():
    """Trains the chesspiece model based on AlexNet."""
    model = alexnet(input_shape=(224, 224, 3))

    train_generator, validation_generator = data_generators(
        preprocess_input, (224, 224), 64)

    callbacks = model_callbacks(20, "./models/AlexNet.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/AlexNet_acc.png",
                       "./models/AlexNet_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/AlexNet_last.h5")


def continue_training():
    """Continues training the chesspiece model based on AlexNet."""
    model = load_model("./models/AlexNet.h5")

    train_generator, validation_generator = data_generators(
        preprocess_input, (224, 224), 64)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = model_callbacks(20, "./models/AlexNet_2.h5", 0.2, 8)

    history = train_model(model, 100, train_generator, validation_generator,
                          callbacks, use_weights=False, workers=5)

    plot_model_history(history, "./models/AlexNet_2_acc.png",
                       "./models/AlexNet_2_loss.png")
    evaluate_model(model, validation_generator)

    model.save("./models/AlexNet_2_last.h5")


if __name__ == "__main__":
    train_chesspiece_model()
    continue_training()
