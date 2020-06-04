"""
Processes input boards and their transformation to individual pieces
using the detectboard module.
"""
import glob
import os
import shutil

import cv2
from tqdm import tqdm

from lc2fen.detectboard import detect_board
from lc2fen.fen import fen_to_board, rotate_board_fen2image


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


def split_square_board_image(board_image, output_name, out_dir, board=None):
    """
    Splits a board image into the 8x8 squares.

    Splits the board_image into individual squares and saves them into
    out_dir as output_name appended with the corresponding position.

    :param board_image: The board image to split. Must be a square in
        size.
    :param output_name: Starting name of the pieces.
    :param out_dir: Output directory where to save the pieces.
    :param board: If board is not None, it represents the 8x8 board
        matrix with the corresponding labels in each position.
    """
    img = cv2.imread(board_image)
    if img.shape[0] != img.shape[1]:
        raise ValueError("Image must be a square in size")
    square_size = img.shape[0] // 8  # 1200 / 8
    for row_start in range(0, img.shape[0], square_size):
        i = row_start // square_size
        for col_start in range(0, img.shape[1], square_size):
            j = col_start // square_size
            if board:
                out_loc = str(out_dir) + "/" + str(board[i][j]) + "/" + str(
                    output_name) + "_" + str(i) + "_" + str(j) + ".jpg"
            else:
                out_loc = str(out_dir) + "/" + str(output_name) + "_" + str(
                    i) + "_" + str(j) + ".jpg"

            cv2.imwrite(out_loc, img[row_start:row_start + square_size,
                                     col_start:col_start + square_size])


def split_board_image(board_image, square_corners, output_name, out_dir,
                      board=None):
    """
    Splits a board into the individual squares using their corners.

    Splits the board_image into the content of individual squares and
    saves them into out_dir as output_name appended with the
    corresponding position.

    :param board_image: The board image to split.
    :param square_corners: List with the coordinates of each of the
        corners of the squares of the chessboard.
    :param output_name: Starting name of the pieces.
    :param out_dir: Output directory where to save the pieces.
    :param board: If board is not None, it represents the 8x8 board
        matrix with the corresponding labels in each position.
    """
    for row_ind in range(1, 9):  # Square_corners is 9x9
        for col_ind in range(0, 8):
            # Coords of each square
            bl_corner = square_corners[row_ind + 9 * col_ind]
            br_corner = square_corners[row_ind + 9 * (col_ind + 1)]
            tl_corner = square_corners[(row_ind - 1) + 9 * col_ind]

            # Output dir and name
            if board:
                out_loc = str(out_dir) + "/" + str(
                    board[row_ind - 1][col_ind]) + "/" + str(
                    output_name) + "_" + str(row_ind - 1) + "_" + str(
                    col_ind) + ".jpg"
            else:
                out_loc = str(out_dir) + "/" + str(output_name) + "_" + str(
                    row_ind - 1) + "_" + str(col_ind) + ".jpg"

            # Height of the image
            height = int((bl_corner[1] - tl_corner[1]) * 1.5)

            # Check if we are outside of the image
            if bl_corner[1] - height < 0 or br_corner[1] - height < 0:
                height = min(bl_corner[1], br_corner[1])

            # Remember, image is [y1:y2, x1:x2, :], y1 < y2, x1 < x2
            rect = board_image[bl_corner[1] - height:bl_corner[1],
                               bl_corner[0]:br_corner[0], :]

            cv2.imwrite(out_loc, rect)


def split_detected_square_boards(data_path):
    """
    Splits all detected .jpg boards in 'data_path/boards/output' into
    individual pieces in 'data_path/pieces' classifying them
    by fen. Line i of .fen file corresponds to board i .jpg. Fen file
    must be in 'data_path/boards/input'.

    :param data_path: Path to the 'data' folder. For example: '../data'.
    """
    detected_boards = sorted(glob.glob(data_path + '/boards/output/*.jpg'))
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
    input_boards = sorted(glob.glob(data_path + '/boards/input/*.jpg'))

    print("INPUT: %d boards" % len(input_boards), flush=True)

    for input_board in tqdm(input_boards):
        output_board = input_board.replace('/input/', '/output/')
        input_image = cv2.imread(input_board)
        image_object = detect_board.detect(input_image, output_board)
        _, square_corners = detect_board.compute_corners(image_object)

        name = os.path.splitext(os.path.basename(input_board))[0]
        split_board_image(input_image, square_corners, name,
                          data_path + '/pieces')
