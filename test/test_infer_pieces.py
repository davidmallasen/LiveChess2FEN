"""This module is responsible for testing "infer_pieces.py" module.

Specifically, it tests the `_determine_changed_squares()` and
`_detect_move()` functions in the module.
"""


from lc2fen.fen import fen_to_board, board_to_list
from lc2fen.infer_pieces import (
    _PIECE_TO_IDX_FULL,
    _determine_changed_squares,
    _detect_move,
)


def generate_probs_with_no_indices_from_fen(fen: str):
    """Generate `probs_with_no_indices` from FEN string.

    This function takes a FEN string as input and generates the
    hardcoded `probs_with_no_indices`.

    :param fen: FEN string of the board position of interest.

    :return: Length-64 list of piece probabilities.

        Each element in the list is a length-13 sublist that corresponds
        to a unique square on the chessboard.

        Each sublist contains 13 piece probabilities (in the order of
        `_IDX_TO_PIECE_FULL`) for the corresponding square.

        The probabilities are hardcoded such that each sublist has exactly
        twelve 0s and one 1, where the 1 corresponds to the piece type given by
        the input FEN string. For example, if `fen` says that there is a piece
        of type `piece` on the a8 square of the chessboard, then the first
        sublist `sublist` of the returned `probs_with_no_indices` will satisfy
        `sublist[_PIECE_TO_IDX_FULL[piece]] == 1`.
    """
    piece_list = board_to_list(fen_to_board(fen))
    probs_with_no_indices = []
    for piece in piece_list:
        sublist: list[float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sublist[_PIECE_TO_IDX_FULL[piece]] = 1

        probs_with_no_indices.append(sublist)

    return probs_with_no_indices


def test_determine_changed_squares_and_detect_move():
    """Test `_determine_changed_squares()` and `_detect_move()`."""
    # Test detection of a white (pawn) move
    previous_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    current_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [35, 51]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (51, 35, "white_moves")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a black (knight) move
    previous_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR"
    current_fen = "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [6, 21]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (6, 21, "black_moves")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a white (pawn) capture
    previous_fen = "r1bqkb1r/ppp2ppp/2n2n2/1N1pp3/3P1B2/8/PPP1PPPP/R2QKBNR"
    current_fen = "r1bqkb1r/ppp2ppp/2n2n2/1N1pP3/5B2/8/PPP1PPPP/R2QKBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [28, 35]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (35, 28, "white_captures")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a black (knight) capture
    previous_fen = "r1bqkb1r/ppp2ppp/2n5/1N1pP2n/5B2/4P3/PPP2PPP/R2QKBNR"
    current_fen = "r1bqkb1r/ppp2ppp/2n5/1N1pP3/5n2/4P3/PPP2PPP/R2QKBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [31, 37]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (31, 37, "black_captures")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a white en passant
    previous_fen = "rnbqkbnr/ppp2ppp/3p4/3Pp3/8/8/PPP1PPPP/RNBQKBNR"
    current_fen = "rnbqkbnr/ppp2ppp/3pP3/8/8/8/PPP1PPPP/RNBQKBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [20, 27, 28]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (27, 20, "white_en_passants")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a black en passant
    previous_fen = "rnbqkbnr/ppp1pppp/8/8/3pP3/3P1N2/PPP2PPP/RNBQKB1R"
    current_fen = "rnbqkbnr/ppp1pppp/8/8/8/3PpN2/PPP2PPP/RNBQKB1R"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [35, 36, 44]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (35, 44, "black_en_passants")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a white kingside castling
    previous_fen = "r1bq1rk1/ppn2ppp/2p1pn2/3p4/1b1P1B2/2NBPN2/PPP2PPP/R2QK2R"
    current_fen = "r1bq1rk1/ppn2ppp/2p1pn2/3p4/1b1P1B2/2NBPN2/PPP2PPP/R2Q1RK1"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [60, 61, 62, 63]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (60, 62, "white_castles_kingside")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a black kingside castling
    previous_fen = "r1bqk2r/ppn2ppp/2p1pn2/3p4/1b1P1B2/2NBPN2/PPP2PPP/R2QK2R"
    current_fen = "r1bq1rk1/ppn2ppp/2p1pn2/3p4/1b1P1B2/2NBPN2/PPP2PPP/R2QK2R"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [4, 5, 6, 7]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (4, 6, "black_castles_kingside")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a white queenside castling
    previous_fen = "rnbq1rk1/ppp1ppbp/3p1np1/8/3PPB2/2N5/PPPQ1PPP/R3KBNR"
    current_fen = "rnbq1rk1/ppp1ppbp/3p1np1/8/3PPB2/2N5/PPPQ1PPP/2KR1BNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [56, 58, 59, 60]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (60, 58, "white_castles_queenside")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )

    # Test detection of a black queenside castling
    previous_fen = "r3kb1r/pp2pppp/2n5/2q2b2/Q4B2/2P5/PP3PPP/R3KBNR"
    current_fen = "2kr1b1r/pp2pppp/2n5/2q2b2/Q4B2/2P5/PP3PPP/R3KBNR"
    probs_with_no_indices = generate_probs_with_no_indices_from_fen(
        current_fen
    )

    changed_squares = [0, 2, 3, 4]
    assert (
        _determine_changed_squares(previous_fen, probs_with_no_indices)
        == changed_squares
    )

    move = (4, 2, "black_castles_queenside")
    assert (
        _detect_move(previous_fen, probs_with_no_indices, changed_squares)
        == move
    )
