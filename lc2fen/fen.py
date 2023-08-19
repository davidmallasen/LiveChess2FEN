"""This module is responsible for FEN-related transformations."""


PIECE_TYPES = ["r", "n", "b", "q", "k", "p", "P", "R", "N", "B", "Q", "K", "_"]


def fen_to_board(fen: str) -> list[list[str]]:
    """Translate a FEN string to a board matrix.

    Note that the FEN string should only contain information of the
    positions of the pieces. Each empty square is represented by a `"_"`
    in the board matrix.

    :param fen: FEN string to translate.

    :return: Board matrix corresponding to the FEN string.
    """
    rows = fen.split(sep="/")

    if len(rows) != 8:
        raise ValueError(f"fen must have 8 rows: {fen}")

    board = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(["_"] * int(char))
            else:
                board_row.append(char)
        if len(board_row) != 8:
            raise ValueError(f"Each fen row must have 8 positions: {fen}")
        board.append(board_row)

    return board


def board_to_fen(board: list[list[str]]) -> str:
    """Translate a board matrix to a FEN string.

    Each empty square must be represented by a `"_"` in the board
    matrix.

    :param board: Board matrix to translate.

    :return: FEN string corresponding to the board matrix.
    """
    fen = []
    for row in board:
        prev_empty = False
        empty = 0
        for square in row:
            if square == "_":
                empty += 1
                prev_empty = True
            else:
                if prev_empty:
                    prev_empty = False
                    fen.append(str(empty))
                    empty = 0
                fen.append(square)

        if prev_empty:
            fen.append(str(empty))

        fen.append("/")

    return "".join(fen[:-1])  # Remove final /


def list_to_board(pieces_list: list[str], a1_pos="BL") -> list[list[str]]:
    """Translate a list of pieces to a board matrix.

    This function translates a list of pieces to an 8x8 board matrix.
    The board matrix is rotated such that the a1 suqare is in the
    bottom-left corner.

    :param pieces_list: List of pieces to translate.

    :param a1_pos: Position of the a1 square of list of pieces.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the list of pieces.

    :return: Board matrix corresponding to the list of pieces.
    """
    if len(pieces_list) != 64:
        raise ValueError("Input pieces list must be of length 64")

    board = [pieces_list[ind : ind + 8] for ind in range(0, 64, 8)]
    board = rotate_board_to_standard_view(board, a1_pos)
    return board


def board_to_list(board: list[list[str]]) -> list[str]:
    """Translate a board matrix to a list of pieces.

    :param board: Board matrix to translate.

    :return: List of pieces corresponding to the board matrix.
    """
    return [pos for row in board for pos in row]


def is_light_square(list_pos: int) -> bool:
    """Determine whether a chess square is a light square or not.

    This function returns `True` if the chess square corresponding to
    `list_pos` is a light square. Otherwise it returns `False`.

    :param list_pos: Integer corresponding to position of chess square.

        `0` corresponds to the a8 square, `1` corresponds to the b8
        square, ..., `63` corresponds to the h1 square.

    :return: Whether the chess square is a light square or not.
    """
    if not 0 <= list_pos <= 63:
        raise ValueError("List position must be between 0 and 63")

    if list_pos % 16 < 8:  # Rank 8, 6, 4, or 2
        return list_pos % 2 == 0  # File a, c, e, or g
    else:  # Rank 7, 5, 3, or 1
        return list_pos % 2 == 1  # File b, d, f, or h


def rotate_board_from_standard_view(
    board: list[list[str]], a1_pos: str
) -> list[list[str]]:
    """Rotate a board matrix whose a1 square is in bottom-left corner.

    :param board: Board matrix whose a1 square is in bottom-left corner.

    :param a1_pos: Position of the a1 square of rotated board matrix.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the rotated board matrix. (B =
        bottom, T = top, R = right, and L = left.)

    :return: Rotated board matrix.
    """
    if a1_pos == "BL":
        return board
    if a1_pos == "BR":  # Counterclockwise rotation
        return list(map(list, zip(*board)))[::-1]
    if a1_pos == "TL":  # Clockwise rotation
        return list(map(list, zip(*board[::-1])))
    if a1_pos == "TR":  # 180 degree rotation
        tmp = list(map(list, zip(*board[::-1])))
        return list(map(list, zip(*tmp[::-1])))

    raise ValueError("a1_pos is not BL, BR, TL or TR")


def rotate_board_to_standard_view(
    board: list[list[str]], a1_pos: str
) -> list[list[str]]:
    """Rotate board matrix s.t. its a1 square ends up in BL corner.

    This function rotates a board matrix such that its a1 square ends up
    in the bottom-left corner.

    :param board: Board matrix whose a1 square is in `a1_pos` corner.

    :param a1_pos: Position of the a1 square of (input) board matrix.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the (input) board matrix. (B =
        bottom, T = top, R = right, and L = left.)

    :return: Rotated board matrix.

        The rotated board matrix has its a1 square in the bottom-left
        corner.
    """
    # In the next two comments, C = clockwise and CC = counterclockwise
    if a1_pos == "BR":
        a1_pos = "TL"  # Exchange 90-degree C and 90-degree CC rotations
    elif a1_pos == "TL":
        a1_pos = "BR"  # Exchange 90-degree C and 90-degree CC rotations

    return rotate_board_from_standard_view(
        board, a1_pos
    )  # The 180-degree rotations stay the same


def compare_fen(fen1: str, fen2: str) -> int:
    """Return the number of positions that differ for two FEN strings.

    :param fen1: First FEN string.

    :param fen2: Second FEN string.

    :return: Number of positions that differ for the two FEN strings.
    """
    board1 = fen_to_board(fen1)
    board2 = fen_to_board(fen2)
    count = 0
    for i in range(8):
        for j in range(8):
            if board1[i][j] != board2[i][j]:
                count += 1
    return count
