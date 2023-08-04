"""This module works with the chess-piece dataset."""


import csv
import functools
import os
import shutil
from random import shuffle

import pandas as pd

from lc2fen.fen import PIECE_TYPES


PIECES_TO_CLASSNUM = {
    "_": 0,
    "b": 1,
    "k": 2,
    "n": 3,
    "p": 4,
    "q": 5,
    "r": 6,
    "B": 7,
    "K": 8,
    "N": 9,
    "P": 10,
    "Q": 11,
    "R": 12,
}


def create_dataset_csv(dataset_dir, csv_name, frac=1, validate=0.2, test=0.1):
    """Create the csv for the dataset.

    Note that this function is deprecated and not currently in use.

    :param dataset_dir: Directory of the dataset.

    :param csv_name: Name of the output csv.

    :param frac: Fraction of images to load.

    :param validate: Fraction of images to label as VAL.

    :param test: Fraction of images to label as TEST.

    :return: Number of loaded images.
    """

    def load_dataset_images(dataset_dir, frac):
        """Return a `DataFrame` with the loaded dataset images.

        :param dataset_dir: Directory of the dataset.

        :param frac: Fraction of images to load.
        """
        file_names = [
            (
                piece_type,
                [
                    dataset_dir + piece_type + "/" + str(x)
                    for x in os.listdir(dataset_dir + piece_type)
                ],
            )
            for piece_type in PIECE_TYPES
        ]

        file_names_label = [
            list(zip(images, [piece_type for x in images]))
            for piece_type, images in file_names
        ]

        data_frame = pd.DataFrame(
            data=functools.reduce(lambda x, y: x + y, file_names_label)
        )
        data_frame = data_frame.rename(columns={0: "image_name", 1: "label"})
        # Shuffle rows
        return data_frame.sample(frac=frac).reset_index(drop=True)

    data_frame = load_dataset_images(dataset_dir, frac)
    total_rows = len(data_frame.index)

    with open(
        dataset_dir + csv_name, "w", newline="", encoding="utf-8"
    ) as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        start_test = 1.0 - test
        start_validate = start_test - validate

        for i, row in data_frame.iterrows():
            percentage = i / total_rows
            set_str = "TRAIN"
            if percentage >= start_test:
                set_str = "TEST"
            elif percentage >= start_validate:
                set_str = "VAL"

            filename, label = row
            label = PIECES_TO_CLASSNUM[label]
            csvwriter.writerow([set_str, filename, label])
    return total_rows


def randomize_dataset(dataset_dir):
    """Randomize the order of images in subdirectories of `dataset_dir`.

    The randomized images are renamed using the "<number>.jpg" format.

    :param dataset_dir: Directory of the dataset.
    """
    dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    for dir in dirs:
        files = os.listdir(dataset_dir + "/" + dir)
        shuffle(files)

        for i, file in enumerate(files):
            path = os.path.join(dataset_dir, dir, file)
            if os.path.isfile(path):
                newpath = os.path.join(dataset_dir, dir, str(i) + ".jpg")
                os.rename(path, newpath)


def split_dataset(dataset_dir, train_dir, validation_dir, train_perc=0.8):
    """Split the full dataset into training and validation datasets.

    This function splits the full dataset (`dataset_dir`) into a
    training dataset (`train_dir`) and a validation dataset
    (`validation_dir`).

    :param dataset_dir: Directory of the whole dataset.

    :param train_dir: Train directory.

    :param validation_dir: Validation directory.

    :param train_perc: Percentage of training images.
    """
    shutil.rmtree(train_dir)
    shutil.rmtree(validation_dir)

    os.mkdir(train_dir)
    os.mkdir(train_dir + "/_/")
    os.mkdir(train_dir + "/r/")
    os.mkdir(train_dir + "/n/")
    os.mkdir(train_dir + "/b/")
    os.mkdir(train_dir + "/q/")
    os.mkdir(train_dir + "/k/")
    os.mkdir(train_dir + "/p/")
    os.mkdir(train_dir + "/R/")
    os.mkdir(train_dir + "/N/")
    os.mkdir(train_dir + "/B/")
    os.mkdir(train_dir + "/Q/")
    os.mkdir(train_dir + "/K/")
    os.mkdir(train_dir + "/P/")

    os.mkdir(validation_dir)
    os.mkdir(validation_dir + "/_/")
    os.mkdir(validation_dir + "/r/")
    os.mkdir(validation_dir + "/n/")
    os.mkdir(validation_dir + "/b/")
    os.mkdir(validation_dir + "/q/")
    os.mkdir(validation_dir + "/k/")
    os.mkdir(validation_dir + "/p/")
    os.mkdir(validation_dir + "/R/")
    os.mkdir(validation_dir + "/N/")
    os.mkdir(validation_dir + "/B/")
    os.mkdir(validation_dir + "/Q/")
    os.mkdir(validation_dir + "/K/")
    os.mkdir(validation_dir + "/P/")

    dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    for dir in dirs:
        files = os.listdir(os.path.join(dataset_dir, dir))
        num_train_files = len(files) * train_perc
        for i, file in enumerate(files):
            path = os.path.join(dataset_dir, dir, file)
            if os.path.isfile(path):
                if i < num_train_files:
                    newpath = os.path.join(train_dir, dir, file)
                else:
                    newpath = os.path.join(validation_dir, dir, file)
                shutil.copy(path, newpath)
