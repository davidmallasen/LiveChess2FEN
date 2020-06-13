"""
Chess pieces inference from the probabilities given by the chess piece
convolutional neural networks.
"""
import numpy as np

from lc2fen.fen import board_to_list, list_to_board, is_white_square

__PREDS_DICT = {0: 'B', 1: 'K', 2: 'N', 3: 'P', 4: 'Q', 5: 'R', 6: '_',
                7: 'b', 8: 'k', 9: 'n', 10: 'p', 11: 'q', 12: 'r'}

__IDX_TO_PIECE = {0: 'B', 1: 'N', 2: 'P', 3: 'Q', 4: 'R',
                  5: 'b', 6: 'n', 7: 'p', 8: 'q', 9: 'r'}


def __sort_pieces_list(_pieces_probs_sort):
    """Returns a list of each piece sorted in descending order."""
    w_bishops = sorted(_pieces_probs_sort, key=lambda prob: prob[0][0],
                       reverse=True)
    w_knights = sorted(_pieces_probs_sort, key=lambda prob: prob[0][2],
                       reverse=True)
    # Pawns can't be in the first or last row
    w_pawns = sorted(_pieces_probs_sort[8:-8], key=lambda prob: prob[0][3],
                     reverse=True)
    w_queens = sorted(_pieces_probs_sort, key=lambda prob: prob[0][4],
                      reverse=True)
    w_rooks = sorted(_pieces_probs_sort, key=lambda prob: prob[0][5],
                     reverse=True)
    b_bishops = sorted(_pieces_probs_sort, key=lambda prob: prob[0][7],
                       reverse=True)
    b_knights = sorted(_pieces_probs_sort, key=lambda prob: prob[0][9],
                       reverse=True)
    # Pawns can't be in the first or last row
    b_pawns = sorted(_pieces_probs_sort[8:-8],
                     key=lambda prob: prob[0][10], reverse=True)
    b_queens = sorted(_pieces_probs_sort, key=lambda prob: prob[0][11],
                      reverse=True)
    b_rooks = sorted(_pieces_probs_sort, key=lambda prob: prob[0][12],
                     reverse=True)
    return [w_bishops, w_knights, w_pawns, w_queens, w_rooks, b_bishops,
            b_knights, b_pawns, b_queens, b_rooks]


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


def infer_chess_pieces(pieces_probs, a1_pos):
    """
    Infers the chess pieces in all of the board based on the given
    probabilities.

    :param pieces_probs: List of the probabilities of each class in each
        position of the chessboard given in FEN notation order.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :return: A list of the inferred chess pieces in FEN notation order.
    """
    pieces_probs = board_to_list(list_to_board(pieces_probs, a1_pos))

    # None represents that no piece is set in that position yet
    out_preds = [None] * 64

    # We need to store the original order
    pieces_probs_sort = [(probs, i) for i, probs in enumerate(pieces_probs)]

    # First choose the kings, there must be one of each color
    white_king = max(pieces_probs_sort, key=lambda prob: prob[0][1])
    black_kings = sorted(pieces_probs_sort, key=lambda prob: prob[0][8],
                         reverse=True)  # Descending order

    black_king = black_kings[0]
    if black_king[1] == white_king[1]:
        black_king = black_kings[1]

    out_preds[white_king[1]] = 'K'
    out_preds[black_king[1]] = 'k'

    out_preds_empty = 62  # We have already set the kings

    # Then set the blank spaces, the CNN has a very high accuracy
    # detecting these cases
    for idx, piece in enumerate(pieces_probs):
        if out_preds[idx] is None:
            pred_idx = np.argmax(piece)
            if __PREDS_DICT[pred_idx] == '_':
                out_preds[idx] = '_'
                out_preds_empty -= 1

    # Save if there is already a bishop in a [white, black] square
    w_bishop_sq = [False, False]
    b_bishop_sq = [False, False]

    # Set the rest of the pieces in the order given by the highest
    # probability of any piece for all the board
    pieces_lists = __sort_pieces_list(pieces_probs_sort)
    # Index to the highest probability, from each list in pieces_lists,
    # that we have not set yet (in the same order than above).
    idx = [0] * 10
    # Top of each sorted piece list (highest probability of each piece)
    tops = [piece_list[0] for piece_list in pieces_lists]
    # Maximum number of pieces of each type in the same order than tops
    max_pieces_left = [2, 2, 8, 9, 2, 2, 2, 8, 9, 2]

    while out_preds_empty > 0:
        # Fill in the square in out_preds that has the piece with the
        # maximum probability of all the board
        max_idx = __max_piece(tops)
        # If we haven't maxed that piece type and the square is empty
        if (max_pieces_left[max_idx] > 0
                and out_preds[tops[max_idx][1]] is None
                and __check_bishop(max_idx, tops, w_bishop_sq, b_bishop_sq)):
            # Fill the square and update counters
            out_preds[tops[max_idx][1]] = __IDX_TO_PIECE[max_idx]
            out_preds_empty -= 1
            max_pieces_left[max_idx] -= 1
        # In any case we must update the entry in tops with the next
        # highest probability for the piece type we have tried
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
    return __PREDS_DICT[np.argmax(square_probs)] == '_'


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
