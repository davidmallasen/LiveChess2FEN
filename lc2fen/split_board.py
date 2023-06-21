"""
Functions for splitting a board into individual pieces. Contains a
trivial split into the 8x8 squares and a more advanced method that takes
into account the piece height and perspective.
"""
import cv2


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
            height = int((bl_corner[1] - tl_corner[1]) * 1.75)

            # Check if we are outside of the image
            if bl_corner[1] - height < 0 or br_corner[1] - height < 0:
                height = min(bl_corner[1], br_corner[1])

            # Remember, image is [y1:y2, x1:x2, :], y1 < y2, x1 < x2
            rect = board_image[bl_corner[1] - height:bl_corner[1],
                               bl_corner[0]:br_corner[0], :]

            cv2.imwrite(out_loc, rect)