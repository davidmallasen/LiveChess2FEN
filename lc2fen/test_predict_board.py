"""
Board prediction testing.
"""
import time

from lc2fen.fen import list_to_board, board_to_fen, compare_fen
from lc2fen.predict_board import detect_input_board, \
    obtain_individual_pieces, infer_chess_pieces


def predict_board(predictions_path, board_name, a1_pos, obtain_pieces_probs):
    """
    Predict the fen notation of a chessboard. Prints the elapsed times.

    The obtain_predictions argument allows us to predict using different
    methods (such as Keras, ONNX or TensorRT models) that may need
    additional context.

    :param predictions_path: Path to the 'predictions' folder.
        For example: '../predictions'.
    :param board_name: Name of the board to predict.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param obtain_pieces_probs: Function which receives a list with the
        path to each piece image in FEN notation order and returns the
        corresponding probabilities of each piece belonging to each
        class as another list.
    :return: Predicted fen string representing the chessboard.
    """
    total_time = 0

    start = time.perf_counter()
    detect_input_board(predictions_path, board_name)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time detecting the input board: {elapsed_time}")

    start = time.perf_counter()
    pieces = obtain_individual_pieces(predictions_path, board_name)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time obtaining the individual pieces: {elapsed_time}")

    start = time.perf_counter()
    pieces_probs = obtain_pieces_probs(pieces)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time predicting probabilities: {elapsed_time}")

    start = time.perf_counter()
    predictions = infer_chess_pieces(pieces_probs, a1_pos)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time inferring chess pieces: {elapsed_time}")

    start = time.perf_counter()
    board = list_to_board(predictions)
    fen = board_to_fen(board)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time converting to fen notation: {elapsed_time}")

    print(f"Elapsed total time: {total_time}")

    return fen


def print_fen_comparison(board_name, fen, correct_fen, n_pieces):
    """
    Compares the predicted fen with the correct fen and pretty prints
    the result.

    :param board_name: Name of the board. For example: 'test1.jpg'
    :param fen: Predicted fen string.
    :param correct_fen: Correct fen string.
    :param n_pieces: Number of pieces in the board.
    """
    n_dif = compare_fen(fen, correct_fen)
    print(board_name[:-4] + ': ' + str(n_dif) + " {:.2f}% ".format(
        1 - (n_dif / 64)) + "{:.2f}% ".format(
        1 - (n_dif / n_pieces)) + fen + '\n')
