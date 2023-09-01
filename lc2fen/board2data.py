"""
This module is responsible for board-to-pieces transformations.

It transforms a chessboard into individual pieces. It is useful for
detecting chessboards (with "board_detection.py") and creating images
for a dataset.
"""


import glob
import os
import re
import shutil

import cv2
from tqdm import tqdm

from lc2fen.detectboard import detect_board
from lc2fen.fen import fen_to_board, rotate_board_from_standard_view
from lc2fen.split_board import (
    split_board_image_trivial,
    split_board_image_advanced,
)


def regenerate_data_folder(data_path: str):
    """Regenerate the data folder.

    This function deletes the "`data_path`/boards/output" folder, the
    "`data_path`/boards/debug_steps" folder, and the
    "`data_path`/pieces" folder, and then it regenerates these folders
    (the folders will be empty) and creates the 13 piece subfolders.

    :param data_path: Path to the data folder. Example: `"data"`.
    """
    shutil.rmtree(data_path + "/boards/output/")
    shutil.rmtree(data_path + "/boards/debug_steps/")
    shutil.rmtree(data_path + "/pieces/")

    os.mkdir(data_path + "/boards/output/")
    os.mkdir(data_path + "/boards/debug_steps/")
    os.mkdir(data_path + "/pieces/")
    os.mkdir(data_path + "/pieces/_/")
    os.mkdir(data_path + "/pieces/r/")
    os.mkdir(data_path + "/pieces/n/")
    os.mkdir(data_path + "/pieces/b/")
    os.mkdir(data_path + "/pieces/q/")
    os.mkdir(data_path + "/pieces/k/")
    os.mkdir(data_path + "/pieces/p/")
    os.mkdir(data_path + "/pieces/R/")
    os.mkdir(data_path + "/pieces/N/")
    os.mkdir(data_path + "/pieces/B/")
    os.mkdir(data_path + "/pieces/Q/")
    os.mkdir(data_path + "/pieces/K/")
    os.mkdir(data_path + "/pieces/P/")


def split_detected_square_boards(data_path: str):
    """Split all the detected boards into individual pieces.

    This function splits all the detected .jpg boards in the
    "`data_path`/boards/output" folder into individual pieces in the
    "`data_path`/pieces" folder. The pieces are appropriately
    classified based on the provided .fen file, which must be in the
    "`data_path`/boards/input" folder.

    Note: line i of .fen file must correspond be the FEN string of the
    board position corresponding to the "board i .jpg" image.

    :param data_path: Path to the data folder. Example: `"data"`.
    """

    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    detected_boards = sorted(
        glob.glob(data_path + "/boards/output/*.jpg"), key=natural_key
    )
    print("DETECTED: %d boards" % len(detected_boards))

    fen_file = glob.glob(data_path + "/boards/input/*.fen")
    if len(fen_file) != 1:
        raise ValueError("Exactly one FEN file must be in the input folder")

    with open(fen_file[0], "r") as fen_fd:
        for detected_board in detected_boards:
            # Next line without '\n' split by spaces
            line = fen_fd.readline()[:-1].split()
            if len(line) != 2:
                raise ValueError(
                    "All lines in the FEN file must have the format "
                    "'fen orientation'"
                )

            board_labels = rotate_board_from_standard_view(
                fen_to_board(line[0]), line[1]
            )
            name = os.path.splitext(os.path.basename(detected_board))[0]
            split_board_image_trivial(
                detected_board, name, data_path + "/pieces", board_labels
            )


def process_input_boards(data_path: str):
    """Detect boards and store the results.

    This function detects all the boards in the
    "`data_path`/boards/input" folder and stores the results in
    the "`data_path`/boards/output" folder.

    :param data_path: Path to the data folder. Example: `"data"`.
    """

    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    input_boards = sorted(
        glob.glob(data_path + "/boards/input/*.jpg"), key=natural_key
    )

    print("INPUT: %d boards" % len(input_boards), flush=True)

    for input_board in tqdm(input_boards):
        output_board = input_board.replace("/input/", "/output/")
        input_image = cv2.imread(input_board)
        image_object = detect_board.detect(input_image, output_board)
        _, square_corners = detect_board.compute_corners(image_object)

        name = os.path.splitext(os.path.basename(input_board))[0]
        split_board_image_advanced(
            input_image, square_corners, name, data_path + "/pieces"
        )
