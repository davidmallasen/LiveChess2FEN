"""This module is responsible for board-into-individual-pieces splits.

Specifically, it contains implementations of a trivial method that
splits a board into the 64 squares and a more advanced method that takes
into account the piece height and perspective.
"""


import cv2


def split_board_image_trivial(
    board_image: str,
    output_name: str,
    out_dir: str,
    board: list[list[str]] = None,
):
    """Split a chessboard image into 64 images of the 64 squares.

    This function splits a `board_image` into individual squares. Each
    individual square is saved into `out_dir` as `output_name` appended
    with its corresponding position on the chessboard.

    For example, if `output_name` is `"square"` and `output_dir` is
    `"Squares"`, then the "Squares" folder will be created in the
    current directory and the image corresponding to the a8 square of
    `board_image` will have the "squares_0_0.jpg" filename and saved
    into the "Squares" folder.

    :param board_image: Chessboard image to split.

        This image's height must be the same as its width.

    :param output_name: Starting name of the 64 output images.

    :param out_dir: Output directory where the output images are saved.

    :param board: 8x8 board matrix that specifies what piece is on each
    square.
    """
    img = cv2.imread(board_image)
    if img.shape[0] != img.shape[1]:
        raise ValueError("Image must have the same height and width.")
    square_size = img.shape[0] // 8  # This is typically 1200 / 8
    for row_start in range(0, img.shape[0], square_size):
        i = row_start // square_size
        for col_start in range(0, img.shape[1], square_size):
            j = col_start // square_size
            if board:
                out_loc = (
                    str(out_dir)
                    + "/"
                    + str(board[i][j])
                    + "/"
                    + str(output_name)
                    + "_"
                    + str(i)
                    + "_"
                    + str(j)
                    + ".jpg"
                )
            else:
                out_loc = (
                    str(out_dir)
                    + "/"
                    + str(output_name)
                    + "_"
                    + str(i)
                    + "_"
                    + str(j)
                    + ".jpg"
                )

            cv2.imwrite(
                out_loc,
                img[
                    row_start : row_start + square_size,
                    col_start : col_start + square_size,
                ],
            )


def split_board_image_advanced(
    board_image: str,
    square_corners: list[tuple[int, int]],
    output_name: str,
    out_dir: str,
    board: list[list[str]] = None,
):
    """Split a board image into 64 square images using square corners.

    This function splits a chessboard image into 64 images of the 64
    squares using the coordinates of the 81 square corners. Each
    individual square is saved into `out_dir` as `output_name` appended
    with its corresponding position on the chessboard.

    For example, if `output_name` is `"square"` and `output_dir` is
    `"Squares"`, then the "Squares" folder will be created in the
    current directory and the image corresponding to the a8 square of
    `board_image` will have the "squares_0_0.jpg" filename and saved
    into the "Squares" folder.

    Note: this function is currently not used in the main workflow of
    the LiveChess2FEN framework (`obtain_individual_pieces()` in
    "predict_board.py" does not use this function). This is because the
    CNNs were trained with a dataset of images that were cropped using
    the trivial method (see `split_detected_square_boards()` in
    "board2data.py") instead of this more advanced one (see
    `process_input_boards()` in "board2data.py"). If the CNNs were
    trained using a dataset whose images are cropped with this advanced
    method, the overall accuracy would improve.

    :param board_image: Chessboard image to split.

    :param square_corners: Length-81 list of square-corner coordinates.

        Each element of the list is a length-2 tuple.

        Each tuple corresponds to a unique square corner on the
        chessboard.

        The tuples are sorted such that the square corners that they
        correspond to progress from top left to bottom right. (If we
        denote the top-left corner as (0, 0) , the top-right corner as
        (0, 8), and the bottom-right corner as (8, 8), then the first
        tuple corresponds to (0, 0), the second corresponds to (0, 1),
        ..., the ninth corresponds to (0, 8), the tenth corresponds to
        (1, 0), ..., the eleventh sorresponds to (1, 8), and so on.)

    :param output_name: Starting name of the 64 output images.

    :param out_dir: Output directory where the output images are saved.

    :param board: 8x8 board matrix that specifies what piece is on each
    square.
    """
    for row_ind in range(1, 9):  # There is a total of 8 rows of squares
        for col_ind in range(0, 8):  # There is a total of 8 columns of squares
            # Extract the coordinates of the bottom-left, bottom-right,
            # and top-left corners
            bl_corner = square_corners[row_ind + 9 * col_ind]
            br_corner = square_corners[row_ind + 9 * (col_ind + 1)]
            tl_corner = square_corners[(row_ind - 1) + 9 * col_ind]

            # Output dir and name
            if board:
                out_loc = (
                    str(out_dir)
                    + "/"
                    + str(board[row_ind - 1][col_ind])
                    + "/"
                    + str(output_name)
                    + "_"
                    + str(row_ind - 1)
                    + "_"
                    + str(col_ind)
                    + ".jpg"
                )
            else:
                out_loc = (
                    str(out_dir)
                    + "/"
                    + str(output_name)
                    + "_"
                    + str(row_ind - 1)
                    + "_"
                    + str(col_ind)
                    + ".jpg"
                )

            # Compute the height of the square image
            height = int((bl_corner[1] - tl_corner[1]) * 1.75)

            # Check if we are outside of the chessboard image
            if bl_corner[1] - height < 0 or br_corner[1] - height < 0:
                height = min(bl_corner[1], br_corner[1])

            # Remember, the square image is [y1:y2, x1:x2, :],
            # where y1 < y2 and x1 < x2
            rect = board_image[
                bl_corner[1] - height : bl_corner[1],
                bl_corner[0] : br_corner[0],
                :,
            ]

            cv2.imwrite(out_loc, rect)
