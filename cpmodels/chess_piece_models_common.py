"""This module has common functions for training chess-piece models."""


import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


def build_model(base_model: Model) -> Model:
    """Build the model from a pretrained base model.

    :param base_model: Base model from keras applications.

        Example: `MobileNetV2(input_shape=(224, 224, 3),
        include_top=False, weights='imagenet')`.

    :return: The compiled model to train.
    """
    layers = base_model.output
    layers = GlobalAveragePooling2D()(layers)
    layers = Dense(1024, activation="relu")(layers)
    preds = Dense(13, activation="softmax")(layers)

    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def data_generators(
    preprocessing_func,
    target_size: tuple[int, int],
    batch_size: int,
    train_path: str = "../data/dataset/train/",
    validation_path: str = "../data/dataset/validation/",
):
    """Return the train and validation generators.

    :param preprocessing_func: Preprocessing function for base model.

        This is the preprocessing function for the pretrained base
        model.

        Example: `from keras.applications.mobilenet_v2 import
        preprocess_input`.

    :param target_size: Dimensions to which all images will be resized.

        Example: `(224, 224)`.

    :param batch_size: Size of the batches of data.

    :param train_path: Path to the train folder.

    :param validation_path: Path to the validation folder.

    :return: Train and validation generators.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func, dtype="float16"
    )

    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        validation_path,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    return train_gen, val_gen


def train_model(
    model: Model,
    epochs: int,
    train_generator,
    val_generator,
    callbacks,
    use_weights: bool,
    workers: int,
):
    """Train the input model."""
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = val_generator.n // val_generator.batch_size

    if use_weights:
        weights = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 0.125,
            4: 1.0,
            5: 1.0,
            6: 0.05,
            7: 1.0,
            8: 1.0,
            9: 1.0,
            10: 0.125,
            11: 1.0,
            12: 1.0,
        }
    else:
        weights = None

    return model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
        class_weight=weights,
        use_multiprocessing=True,
        workers=workers,
    )


def model_callbacks(
    early_stopping_patience: int,
    model_checkpoint_dir,
    reducelr_factor: float,
    reducelr_patience: int,
) -> list:
    """Initialize the model callbacks."""
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        patience=early_stopping_patience,
        restore_best_weights=True,
        min_delta=0.002,
    )
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_dir,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=reducelr_factor,
        patience=reducelr_patience,
        verbose=1,
    )
    return [early_stopping, model_checkpoint, reduce_lr]


def plot_model_history(history, accuracy_savedir, loss_savedir):
    """Plot the model history (accuracy and loss)."""
    # Summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(accuracy_savedir)
    plt.close()

    # Summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"])
    plt.savefig(loss_savedir)
    plt.close()


def evaluate_model(model, test_generator):
    """
    Print the test loss and accuracy of the model.

    :param model: Model to evaluate.

    :param test_generator: Generator with which to test the model.
    """
    scores = model.evaluate(test_generator, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])
