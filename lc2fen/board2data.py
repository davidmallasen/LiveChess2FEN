"""
Processes input boards and their transformation to individual pieces.
Useful when testing the detectboard module or to create images for a
dataset.
"""
import glob
import os
import re
import shutil

import cv2
from tqdm import tqdm

from lc2fen.detectboard import detect_board
from lc2fen.fen import fen_to_board, rotate_board_fen2image
from lc2fen.split_board import split_square_board_image, split_board_image


def regenerate_data_state(data_path):
    """
    Regenerates the state of the data directory.

    Deletes all files and subdirectories in 'data_path/boards/output',
    'data_path/boards/debug_steps' and 'data_path/pieces'. Regenerates
    these directories and the pieces subdirectories.

    :param data_path: Path to the 'data' folder. For example: '../data'.
    """

    shutil.rmtree(data_path + '/boards/output/')
    shutil.rmtree(data_path + '/boards/debug_steps/')
    shutil.rmtree(data_path + '/pieces/')

    os.mkdir(data_path + '/boards/output/')
    os.mkdir(data_path + '/boards/debug_steps/')
    os.mkdir(data_path + '/pieces/')
    os.mkdir(data_path + '/pieces/_/')
    os.mkdir(data_path + '/pieces/r/')
    os.mkdir(data_path + '/pieces/n/')
    os.mkdir(data_path + '/pieces/b/')
    os.mkdir(data_path + '/pieces/q/')
    os.mkdir(data_path + '/pieces/k/')
    os.mkdir(data_path + '/pieces/p/')
    os.mkdir(data_path + '/pieces/R/')
    os.mkdir(data_path + '/pieces/N/')
    os.mkdir(data_path + '/pieces/B/')
    os.mkdir(data_path + '/pieces/Q/')
    os.mkdir(data_path + '/pieces/K/')
    os.mkdir(data_path + '/pieces/P/')


def split_detected_square_boards(data_path):
    """
    Splits all detected .jpg boards in 'data_path/boards/output' into
    individual pieces in 'data_path/pieces' classifying them
    by fen. Line i of .fen file corresponds to board i .jpg. Fen file
    must be in 'data_path/boards/input'.

    :param data_path: Path to the 'data' folder. For example: '../data'.
    """
    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    detected_boards = sorted(glob.glob(data_path + '/boards/output/*.jpg'),
                             key=natural_key)
    print("DETECTED: %d boards" % len(detected_boards))

    fen_file = glob.glob(data_path + '/boards/input/*.fen')
    if len(fen_file) != 1:
        raise ValueError("Only one fen file must be in input directory")

    with open(fen_file[0], 'r') as fen_fd:
        for detected_board in detected_boards:
            # Next line without '\n' split by spaces
            line = fen_fd.readline()[:-1].split()
            if len(line) != 2:
                raise ValueError("All lines in fen file must have the format "
                                 "'fen orientation'")

            board_labels = rotate_board_fen2image(fen_to_board(line[0]),
                                                  line[1])
            name = os.path.splitext(os.path.basename(detected_board))[0]
            split_square_board_image(detected_board, name,
                                     data_path + '/pieces', board_labels)


def process_input_boards(data_path):
    """
    Detects all boards in 'data_path/boards/input' and stores the
    results in 'data_path/boards/output'.

    :param data_path: Path to the 'data' folder. For example: '../data'.
    """
    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    input_boards = sorted(glob.glob(data_path + '/boards/input/*.jpg'),
                          key=natural_key)

    print("INPUT: %d boards" % len(input_boards), flush=True)

    for input_board in tqdm(input_boards):
        output_board = input_board.replace('/input/', '/output/')
        input_image = cv2.imread(input_board)
        image_object = detect_board.detect(input_image, output_board)
        _, square_corners = detect_board.compute_corners(image_object)

        name = os.path.splitext(os.path.basename(input_board))[0]
        split_board_image(input_image, square_corners, name,
                          data_path + '/pieces')
