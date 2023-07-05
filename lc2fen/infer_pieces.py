"""
Chess pieces inference from the probabilities given by the chess piece
convolutional neural networks.
"""
import numpy as np
import chess

from lc2fen.fen import board_to_list, list_to_board, is_white_square, fen_to_board

__PREDS_DICT = {
    0: "B",
    1: "K",
    2: "N",
    3: "P",
    4: "Q",
    5: "R",
    6: "_",
    7: "b",
    8: "k",
    9: "n",
    10: "p",
    11: "q",
    12: "r",
}

__IDX_TO_PIECE = {
    0: "B",
    1: "N",
    2: "P",
    3: "Q",
    4: "R",
    5: "b",
    6: "n",
    7: "p",
    8: "q",
    9: "r",
}

__WHITE_PIECES = ("P", "B", "N", "R", "K", "Q")
__BLACK_PIECES = ("p", "b", "n", "r", "k", "q")

FILES = "abcdefgh"
RANKS = "87654321"


def __sort_pieces_list(_pieces_probs_sort):
    """Returns a list of each piece sorted in descending order."""
    w_bishops = sorted(_pieces_probs_sort, key=lambda prob: prob[0][0], reverse=True)
    w_knights = sorted(_pieces_probs_sort, key=lambda prob: prob[0][2], reverse=True)
    # Pawns can't be in the first or last row
    w_pawns = sorted(
        _pieces_probs_sort[8:-8], key=lambda prob: prob[0][3], reverse=True
    )
    w_queens = sorted(_pieces_probs_sort, key=lambda prob: prob[0][4], reverse=True)
    w_rooks = sorted(_pieces_probs_sort, key=lambda prob: prob[0][5], reverse=True)
    b_bishops = sorted(_pieces_probs_sort, key=lambda prob: prob[0][7], reverse=True)
    b_knights = sorted(_pieces_probs_sort, key=lambda prob: prob[0][9], reverse=True)
    # Pawns can't be in the first or last row
    b_pawns = sorted(
        _pieces_probs_sort[8:-8], key=lambda prob: prob[0][10], reverse=True
    )
    b_queens = sorted(_pieces_probs_sort, key=lambda prob: prob[0][11], reverse=True)
    b_rooks = sorted(_pieces_probs_sort, key=lambda prob: prob[0][12], reverse=True)
    return [
        w_bishops,
        w_knights,
        w_pawns,
        w_queens,
        w_rooks,
        b_bishops,
        b_knights,
        b_pawns,
        b_queens,
        b_rooks,
    ]


def __max_piece(tops):
    """Returns the index of the piece with max probability."""
    value = tops[0][0][0]  # B
    idx = 0
    if tops[1][0][2] > value:  # N
        value = tops[1][0][2]
        idx = 1
    if tops[2][0][3] > value:  # P
        value = tops[2][0][3]
        idx = 2
    if tops[3][0][4] > value:  # Q
        value = tops[3][0][4]
        idx = 3
    if tops[4][0][5] > value:  # R
        value = tops[4][0][5]
        idx = 4
    if tops[5][0][7] > value:  # b
        value = tops[5][0][7]
        idx = 5
    if tops[6][0][9] > value:  # n
        value = tops[6][0][9]
        idx = 6
    if tops[7][0][10] > value:  # p
        value = tops[7][0][10]
        idx = 7
    if tops[8][0][11] > value:  # q
        value = tops[8][0][11]
        idx = 8
    if tops[9][0][12] > value:  # r
        # value = tops[9][0][12]
        idx = 9
    return idx


def __check_bishop(max_idx, tops, w_bishop_sq, b_bishop_sq):
    """
    Checks the position of a bishop. There can be at most one in each
    square color. Returns True if max_idx doesn't represent a bishop. If
    it does, returns if the bishop can be placed in that position.

    Note: this function is no longer used in the code because theoretically,
    for either side, there can be two bishops of the same color (via pawn
    promotion). Since the `max_pieces_left` variable makes sure there are at
    most two bishops for either side, there is no more additional check that
    we need to do for bishops; we can safely remove the requirement that if
    any side (white or black) has two bishops, those two bishops must be
    opposite-colored.
    """
    # If it is a bishop, check that there is at most one in each
    # square color
    if max_idx == 0:  # White bishop
        if is_white_square(tops[max_idx][1]):
            if not w_bishop_sq[0]:
                # We are going to set a white bishop in a white
                # square
                w_bishop_sq[0] = True
                return True
            return False
        if not w_bishop_sq[1]:
            # We are going to set a white bishop in a black square
            w_bishop_sq[1] = True
            return True
        return False
    elif max_idx == 5:  # Black bishop
        if is_white_square(tops[max_idx][1]):
            if not b_bishop_sq[0]:
                # We are going to set a black bishop in a white
                # square
                b_bishop_sq[0] = True
                return True
            return False
        if not b_bishop_sq[1]:
            # We are going to set a white bishop in a black square
            b_bishop_sq[1] = True
            return True
        return False

    return True  # If it's not a bishop, nothing to check


def __determine_promoted_piece(previous_fen, pieces_probs, final_move_sq, color):
    """Determines the promoted piece."""
    promoted_piece_prob = 0
    if color == "white":
        if (
            pieces_probs[final_move_sq][4] > promoted_piece_prob
            and previous_fen.count("Q") < 2
        ):
            promoted_piece = "Q"
            promoted_piece_prob = pieces_probs[final_move_sq][4]
        if (
            pieces_probs[final_move_sq][2] > promoted_piece_prob
            and previous_fen.count("N") < 2
        ):
            promoted_piece = "N"
            promoted_piece_prob = pieces_probs[final_move_sq][2]
        if (
            pieces_probs[final_move_sq][5] > promoted_piece_prob
            and previous_fen.count("R") < 2
        ):
            promoted_piece = "R"
            promoted_piece_prob = pieces_probs[final_move_sq][5]
        if (
            pieces_probs[final_move_sq][0] > promoted_piece_prob
            and previous_fen.count("B") < 2
        ):
            promoted_piece = "B"
    else:
        if (
            pieces_probs[final_move_sq][11] > promoted_piece_prob
            and previous_fen.count("q") < 2
        ):
            promoted_piece = "q"
            promoted_piece_prob = pieces_probs[final_move_sq][11]
        if (
            pieces_probs[final_move_sq][9] > promoted_piece_prob
            and previous_fen.count("n") < 2
        ):
            promoted_piece = "n"
            promoted_piece_prob = pieces_probs[final_move_sq][9]
        if (
            pieces_probs[final_move_sq][12] > promoted_piece_prob
            and previous_fen.count("r") < 2
        ):
            promoted_piece = "r"
            promoted_piece_prob = pieces_probs[final_move_sq][12]
        if (
            pieces_probs[final_move_sq][7] > promoted_piece_prob
            and previous_fen.count("b") < 2
        ):
            promoted_piece = "b"
    # Note that if the provided previous FEN is correct, `promoted_piece`
    # should be defined at this point
    return promoted_piece


def infer_chess_pieces(pieces_probs, a1_pos, previous_fen=None):
    """
    Infers the chess pieces in all of the board based on the given
    probabilities.

    :param pieces_probs: List of the probabilities of each class in each
        position of the chessboard given in FEN notation order.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param previous_fen: FEN string of the previous board position.
        If it is not None, improves piece inference.
    :return: A list of the inferred chess pieces in FEN notation order.
    """
    if previous_fen is not None:
        previous_board = chess.Board(previous_fen)
        previous_list = board_to_list(fen_to_board(previous_fen))
    pieces_probs = board_to_list(list_to_board(pieces_probs, a1_pos))

    # None represents that no piece is set in that position yet
    out_preds = [None] * 64

    final_move_sq = -1
    if previous_fen is not None:  # Perform move detection
        changed_squares_idx = changed_squares(previous_fen, pieces_probs)
        move = inferred_move(previous_fen, pieces_probs, changed_squares_idx)
        if (
            move is not None
        ):  # A move has been successfully detected so the FEN will be concluded immediately
            initial_sq, final_move_sq, action = move
            initial_coordinates = FILES[initial_sq % 8] + RANKS[initial_sq // 8]
            final_coordinates = FILES[final_move_sq % 8] + RANKS[final_move_sq // 8]
            move_UCI = initial_coordinates + final_coordinates
            if action.startswith("white"):
                previous_board.turn = chess.WHITE
            else:
                previous_board.turn = chess.BLACK
            if (
                previous_list[initial_sq] == "P" and initial_coordinates[1] == "7"
            ):  # White promotes (and we have to figure out the promoted piece)
                promoted_piece = __determine_promoted_piece(
                    previous_fen, pieces_probs, final_move_sq, "white"
                )
                move_UCI = move_UCI + promoted_piece.lower()
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))
            elif (
                previous_list[initial_sq] == "p" and initial_coordinates[1] == "2"
            ):  # Black promotes (and we have to figure out the promoted piece)
                promoted_piece = __determine_promoted_piece(
                    previous_fen, pieces_probs, final_move_sq, "black"
                )
                move_UCI = move_UCI + promoted_piece
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))
            elif action.endswith("en_passants"):
                previous_board.ep_square = chess.parse_square(final_coordinates)
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))
            elif action.startswith("white") and action[6:13] == "castles":
                if action.endswith("kingside"):
                    previous_board.set_castling_fen("K")
                else:
                    previous_board.set_castling_fen("Q")
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))
            elif action.startswith("black") and action[6:13] == "castles":
                if action.endswith("kingside"):
                    previous_board.set_castling_fen("k")
                else:
                    previous_board.set_castling_fen("q")
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))
            else:
                previous_board.push_uci(move_UCI)
                return board_to_list(fen_to_board(previous_board.board_fen()))

    # Move detection was either not invoked or not successful, so the pieces on the
    # board will now be inferred one at a time
    pieces_probs_sort = [(probs, i) for i, probs in enumerate(pieces_probs)]

    # First determine the locations of the kings (one white king and one black king)
    white_king = max(pieces_probs_sort, key=lambda prob: prob[0][1])
    black_kings = sorted(
        pieces_probs_sort, key=lambda prob: prob[0][8], reverse=True
    )  # Descending order

    black_king = black_kings[0]
    if black_king[1] == white_king[1]:
        black_king = black_kings[1]

    out_preds[white_king[1]] = "K"
    out_preds[black_king[1]] = "k"

    num_of_undetermined_squares = 62  # We have already determined the king locations

    # Then identify the empty squares (the CNN has a very high accuracy of
    # detecting empty squares)
    for idx, piece in enumerate(pieces_probs):
        if out_preds[idx] is None:
            if is_empty_square(piece):
                out_preds[idx] = "_"
                num_of_undetermined_squares -= 1

    # Determine the locations of the other pieces in order of probability
    # (there is a total of (`num_of_undetermined_squares` * 10) probabilities)
    pieces_lists = __sort_pieces_list(pieces_probs_sort)
    # Keep track of the indices to the squares, whose piece types have not been
    # determined, with the highest probabilities in `pieces_lists` (there are 10
    # piece types left, so we need to keep track of 10 indices)
    idx = [0] * 10
    # Keep track of the top entry of each sorted piece list (corresponding to
    # the square with the highest probability)
    tops = [piece_list[0] for piece_list in pieces_lists]
    # Maximum number of pieces of each type in the same order as `tops`
    max_pieces_left = [2, 2, 8, 2, 2, 2, 2, 8, 2, 2]

    while num_of_undetermined_squares > 0:
        # Determine the piece type of the square that has the piece with the
        # highest probability across the entire board
        max_idx = __max_piece(tops)
        square = tops[max_idx][1]
        # If we haven't maxed that piece type and the piece type of that square
        # hasn't been determined, then we conclude that that square has exactly
        # that piece
        if max_pieces_left[max_idx] > 0 and out_preds[square] is None:
            out_preds[square] = __IDX_TO_PIECE[max_idx]
            num_of_undetermined_squares -= 1
            max_pieces_left[max_idx] -= 1
        # In any case, for the piece type we have tried above, we must replace
        # the entry in `tops` with the next-highest-probability entry
        idx[max_idx] += 1
        tops[max_idx] = pieces_lists[max_idx][idx[max_idx]]

    return out_preds


def is_empty_square(square_probs):
    """
    Infers if the square given by square_probs is empty or not.

    :param square_probs: List of the probabilities of each class in a
        square of the chessboard.
    :return: True if the square_probs infer that the square is empty.
    """
    return __PREDS_DICT[np.argmax(square_probs)] == "_"


def is_white_piece(square_probs):
    """
    Infers if the square given by square_probs contains a white piece.
    This function doesn't check if the square is empty or not, only non-
    empty squares should be tested.

    :param square_probs: List of the probabilities of each class in a
        square of the chessboard.
    :return: True if the square_probs infer that the square contains a
        white piece.
    """
    return np.sum(square_probs[:6]) >= np.sum(square_probs[7:])


def changed_squares(previous_fen, current_probs):
    """
    Checks the squares in which there has been a significant state
    (white, black or empty) change between the last board and the
    current one.

    :param previous_fen: FEN string of the previous board position.
    :param current_probs: List of the probabilities of each class in
        each position of the current chessboard given in FEN notation
        order.
    :return: A list of the indexes of the pieces_probs list indicating
        the positions in which there has been a significant state
        change.
    """
    previous_list = board_to_list(fen_to_board(previous_fen))
    changed_squares_idx = []
    for idx, previous_piece in enumerate(previous_list):
        # Pass the squares in which the previous state (white, black or
        # empty) is the same as the current state
        if previous_piece == "_" and is_empty_square(current_probs[idx]):
            continue
        if (
            previous_piece in __WHITE_PIECES
            and not is_empty_square(current_probs[idx])
            and is_white_piece(current_probs[idx])
        ):
            continue
        if (
            previous_piece in __BLACK_PIECES
            and not is_empty_square(current_probs[idx])
            and not is_white_piece(current_probs[idx])
        ):
            continue
        # If the state has changed
        changed_squares_idx.append(idx)

    return changed_squares_idx


def inferred_move(previous_fen, current_probs, changed_squares_idx):
    """
    Infers the move made. If it can't recognize the move, returns None.

    The inferred action is one of the following: 'white_moves',
    'white_captures', 'black_moves', 'black_captures', 'white_en_passants',
    'black_en_passants', 'white_castles_kingside', 'white_castles_queenside',
    'black_castles_kingside', and 'black_castles_queenside'.

    :param previous_fen: FEN string representing the previous board
        layout.
    :param current_probs: List of the probabilities of each class in
        each position of the current chessboard given in FEN notation
        order.
    :param changed_squares_idx: A list of the indexes of the
        pieces_probs list indicating the positions in which there has
        been a significant state change.
    :return: If it can infer the move, returns a triplet containing the
        index of the initial square, the index of the final square and
        the inferred action. If not, returns None.
    """
    previous_list = board_to_list(fen_to_board(previous_fen))

    if len(changed_squares_idx) == 2:
        # Determine which square is the initial and which is the final
        if is_empty_square(current_probs[changed_squares_idx[0]]):
            initial_sq = changed_squares_idx[0]
            if not is_empty_square(current_probs[changed_squares_idx[1]]):
                final_sq = changed_squares_idx[1]
            else:
                return None
        elif is_empty_square(current_probs[changed_squares_idx[1]]):
            initial_sq = changed_squares_idx[1]
            if not is_empty_square(current_probs[changed_squares_idx[0]]):
                final_sq = changed_squares_idx[0]
            else:
                return None
        else:
            return None

        # We know that in the previous board, the initial square was
        # occupied (now it is empty) and in the current board the final
        # square is occupied
        if previous_list[initial_sq] in __WHITE_PIECES:
            if previous_list[final_sq] == "_":
                if is_white_piece(current_probs[final_sq]):
                    action = "white_moves"
                    return initial_sq, final_sq, action
                else:
                    return None  # White piece converts into a black piece?
            elif previous_list[final_sq] in __BLACK_PIECES:
                if is_white_piece(current_probs[final_sq]):
                    action = "white_captures"
                    return initial_sq, final_sq, action
                else:
                    return None  # White piece converts into a black piece?
            else:
                return None  # White piece captures white piece?
        else:  # The initial square is a black piece
            if previous_list[final_sq] == "_":
                if not is_white_piece(current_probs[final_sq]):
                    action = "black_moves"
                    return initial_sq, final_sq, action
                else:
                    return None  # Black piece converts into a white piece?
            elif previous_list[final_sq] in __WHITE_PIECES:
                if not is_white_piece(current_probs[final_sq]):
                    action = "black_captures"
                    return initial_sq, final_sq, action
                else:
                    return None  # Black piece converts into a white piece?
            else:
                return None  # Black piece captures black piece?

    elif len(changed_squares_idx) == 3:  # En passant
        # Determine the initial square, the final square, and the third square
        if not is_empty_square(current_probs[changed_squares_idx[0]]):
            final_sq = changed_squares_idx[0]
            if previous_list[changed_squares_idx[1]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[1]
                third_sq = changed_squares_idx[2]
            elif previous_list[changed_squares_idx[1]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[1]
                third_sq = changed_squares_idx[2]
            elif previous_list[changed_squares_idx[2]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[2]
                third_sq = changed_squares_idx[1]
            elif previous_list[changed_squares_idx[2]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[2]
                third_sq = changed_squares_idx[1]
            else:
                return None
        elif not is_empty_square(current_probs[changed_squares_idx[1]]):
            final_sq = changed_squares_idx[1]
            if previous_list[changed_squares_idx[0]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[0]
                third_sq = changed_squares_idx[2]
            elif previous_list[changed_squares_idx[0]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[0]
                third_sq = changed_squares_idx[2]
            elif previous_list[changed_squares_idx[2]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[2]
                third_sq = changed_squares_idx[0]
            elif previous_list[changed_squares_idx[2]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[2]
                third_sq = changed_squares_idx[0]
            else:
                return None
        elif not is_empty_square(current_probs[changed_squares_idx[2]]):
            final_sq = changed_squares_idx[2]
            if previous_list[changed_squares_idx[0]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[0]
                third_sq = changed_squares_idx[1]
            elif previous_list[changed_squares_idx[0]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[0]
                third_sq = changed_squares_idx[1]
            elif previous_list[changed_squares_idx[1]] == "P" and is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[1]
                third_sq = changed_squares_idx[0]
            elif previous_list[changed_squares_idx[1]] == "p" and not is_white_piece(
                current_probs[final_sq]
            ):
                initial_sq = changed_squares_idx[1]
                third_sq = changed_squares_idx[0]
            else:
                return None
        else:
            return None

        # Determine the action
        if previous_list[initial_sq] == "P" and previous_list[third_sq] == "p":
            action = "white_en_passants"
            return initial_sq, final_sq, action
        elif previous_list[initial_sq] == "p" and previous_list[third_sq] == "P":
            action = "black_en_passants"
            return initial_sq, final_sq, action
        else:
            return None

    elif len(changed_squares_idx) == 4:  # Castling
        # Determine which square is the initial and which is the final
        # (the initial and final squares of the king, not the rook,
        #  as per the UCI notation)

        if previous_list[changed_squares_idx[0]] in ["K", "k"]:
            initial_sq = changed_squares_idx[0]
            if (
                previous_list[changed_squares_idx[1]] == "_"
                and abs(changed_squares_idx[1] - changed_squares_idx[0]) == 2
            ):
                final_sq = changed_squares_idx[1]
            elif (
                previous_list[changed_squares_idx[2]] == "_"
                and abs(changed_squares_idx[2] - changed_squares_idx[0]) == 2
            ):
                final_sq = changed_squares_idx[2]
            elif (
                previous_list[changed_squares_idx[3]] == "_"
                and abs(changed_squares_idx[3] - changed_squares_idx[0]) == 2
            ):
                final_sq = changed_squares_idx[3]
            else:
                return None
        elif previous_list[changed_squares_idx[1]] in ["K", "k"]:
            initial_sq = changed_squares_idx[1]
            if (
                previous_list[changed_squares_idx[0]] == "_"
                and abs(changed_squares_idx[0] - changed_squares_idx[1]) == 2
            ):
                final_sq = changed_squares_idx[0]
            elif (
                previous_list[changed_squares_idx[2]] == "_"
                and abs(changed_squares_idx[2] - changed_squares_idx[1]) == 2
            ):
                final_sq = changed_squares_idx[2]
            elif (
                previous_list[changed_squares_idx[3]] == "_"
                and abs(changed_squares_idx[3] - changed_squares_idx[1]) == 2
            ):
                final_sq = changed_squares_idx[3]
            else:
                return None
        elif previous_list[changed_squares_idx[2]] in ["K", "k"]:
            initial_sq = changed_squares_idx[2]
            if (
                previous_list[changed_squares_idx[0]] == "_"
                and abs(changed_squares_idx[0] - changed_squares_idx[2]) == 2
            ):
                final_sq = changed_squares_idx[0]
            elif (
                previous_list[changed_squares_idx[1]] == "_"
                and abs(changed_squares_idx[1] - changed_squares_idx[2]) == 2
            ):
                final_sq = changed_squares_idx[1]
            elif (
                previous_list[changed_squares_idx[3]] == "_"
                and abs(changed_squares_idx[3] - changed_squares_idx[2]) == 2
            ):
                final_sq = changed_squares_idx[3]
            else:
                return None
        elif previous_list[changed_squares_idx[3]] in ["K", "k"]:
            initial_sq = changed_squares_idx[3]
            if (
                previous_list[changed_squares_idx[0]] == "_"
                and abs(changed_squares_idx[0] - changed_squares_idx[3]) == 2
            ):
                final_sq = changed_squares_idx[0]
            elif (
                previous_list[changed_squares_idx[1]] == "_"
                and abs(changed_squares_idx[1] - changed_squares_idx[3]) == 2
            ):
                final_sq = changed_squares_idx[1]
            elif (
                previous_list[changed_squares_idx[2]] == "_"
                and abs(changed_squares_idx[2] - changed_squares_idx[3]) == 2
            ):
                final_sq = changed_squares_idx[2]
            else:
                return None
        else:
            return None

        # Determine the action
        if previous_list[initial_sq] == "K" and final_sq == 62:
            action = "white_castles_kingside"
            return initial_sq, final_sq, action
        elif previous_list[initial_sq] == "K" and final_sq == 58:
            action = "white_castles_queenside"
            return initial_sq, final_sq, action
        elif previous_list[initial_sq] == "k" and final_sq == 6:
            action = "black_castles_kingside"
            return initial_sq, final_sq, action
        elif previous_list[initial_sq] == "k" and final_sq == 2:
            action = "black_castles_queenside"
            return initial_sq, final_sq, action
        else:
            return None

    else:  # not len(changed_squares_idx) in [2, 3, 4]
        return None


def __is_king_move(initial_sq, final_sq):
    """At most distance one in any direction."""
    return (
        abs(initial_sq[0] - final_sq[0]) <= 1 and abs(initial_sq[1] - final_sq[1]) <= 1
    )


def __is_rook_move(initial_sq, final_sq):
    """Same row or column."""
    return initial_sq[0] == final_sq[0] or initial_sq[1] == final_sq[1]


def __is_bishop_move(initial_sq, final_sq):
    """Same diagonal."""
    # Parallel to main diagonal
    return (
        initial_sq[0] - initial_sq[1] == final_sq[0] - final_sq[1]
        # Parallel to secondary diagonal
        or initial_sq[0] + initial_sq[1] == final_sq[0] + final_sq[1]
    )


def __is_knight_move(initial_sq, final_sq):
    """L shape."""
    # Row and column distances
    row_d = abs(initial_sq[0] - final_sq[0])
    col_d = abs(initial_sq[1] - final_sq[1])
    return (row_d == 1 and col_d == 2) or (row_d == 2 and col_d == 1)


def __is_pawn_move(initial_sq, final_sq, capturing, white):
    """
    Moves forward in the same column at distance one (or two if it
    hasn't moved yet) and captures forward diagonally at distance one.
    """
    if white:
        if capturing:
            return (
                initial_sq[0] - final_sq[0] == 1
                and abs(initial_sq[1] - final_sq[1]) == 1
            )
        else:
            return initial_sq[1] == final_sq[1] and (
                initial_sq[0] - final_sq[0] == 1
                or (initial_sq[0] - final_sq[0] == 2 and initial_sq[0] == 6)
            )
    else:  # black
        if capturing:
            return (
                initial_sq[0] - final_sq[0] == -1
                and abs(initial_sq[1] - final_sq[1]) == 1
            )
        else:
            return initial_sq[1] == final_sq[1] and (
                initial_sq[0] - final_sq[0] == -1
                or (initial_sq[0] - final_sq[0] == -2 and initial_sq[0] == 1)
            )


def inferred_pieces_from_move(initial_sq, final_sq, action):
    """
    Infers the possible piece types that will occupy the final square
    from the move made.

    Note: since the conclude-fen-immediately-after-move-detection feature
    has been added, this function is no longer used in the code.

    :param initial_sq: Initial square (0-63). As given by inferred_move.
    :param final_sq: Final square (0-63). As given by inferred_move.
    :param action: Action done. As given by inferred_move.
    :return: A list of the unique possible piece types.
    """
    initial_sq = (initial_sq // 8, initial_sq % 8)  # (row, column)
    final_sq = (final_sq // 8, final_sq % 8)

    capturing = action.endswith("captures") | action.endswith("en_passants")
    white = action.startswith("white")
    castling = action[6:13] == "castles"

    possible_pieces = []  # There can't be duplicates

    if white:
        if castling:
            possible_pieces.append("K")
            return possible_pieces

        if __is_pawn_move(initial_sq, final_sq, capturing, white):
            if final_sq[0] == 0:
                # If the move ends in the last row, promotions apply,
                # so the result no longer is a pawn. This move also
                # corresponds with a king, so the result can be all
                # pieces except for the pawn. In this case we don't need
                # to check the rest of the pieces.
                return ["K", "R", "B", "Q", "N"]
            possible_pieces.append("P")
        if __is_king_move(initial_sq, final_sq):
            possible_pieces.append("K")
        if __is_rook_move(initial_sq, final_sq):
            possible_pieces.append("R")
            possible_pieces.append("Q")
        if __is_bishop_move(initial_sq, final_sq):
            possible_pieces.append("B")
            # Bishop and rook moves are exclusive, so Q is not in
            # possible pieces
            possible_pieces.append("Q")
        if __is_knight_move(initial_sq, final_sq):
            possible_pieces.append("N")
    else:  # black
        if castling:
            possible_pieces.append("k")
            return possible_pieces

        if __is_pawn_move(initial_sq, final_sq, capturing, white):
            if final_sq[0] == 7:
                return ["k", "r", "b", "q", "n"]
            possible_pieces.append("p")
        if __is_king_move(initial_sq, final_sq):
            possible_pieces.append("k")
        if __is_rook_move(initial_sq, final_sq):
            possible_pieces.append("r")
            possible_pieces.append("q")
        if __is_bishop_move(initial_sq, final_sq):
            possible_pieces.append("b")
            possible_pieces.append("q")
        if __is_knight_move(initial_sq, final_sq):
            possible_pieces.append("n")
    return possible_pieces
