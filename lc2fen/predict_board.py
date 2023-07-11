"""
Predicts board configurations from images.
"""
import glob
import os
import re
import shutil
import time

import cv2
import numpy as np
import onnxruntime
from keras.models import load_model
from keras.utils.image_utils import load_img, img_to_array
import chess

try:
    import pycuda.driver as cuda

    # pycuda.autoinit causes pycuda to automatically manage CUDA context
    # creation and cleanup.
    import pycuda.autoinit
    import tensorrt as trt
except ImportError:
    cuda = None
    trt = None

from lc2fen.detectboard.detect_board import detect, compute_corners
from lc2fen.fen import (
    list_to_board,
    board_to_fen,
    compare_fen,
    is_white_square,
    fen_to_board,
    board_to_list,
)
from lc2fen.infer_pieces import infer_chess_pieces
from lc2fen.split_board import split_square_board_image


def load_image(img_path, img_size, preprocess_func):
    """
    Loads an image from its path. Intended to use with piece images.

    :param img_path: Image path.
    :param img_size: Size of the input image. For example: 224
    :param preprocess_func: Preprocessing fuction to apply to the input
        image.
    :return: The loaded image.
    """
    img = load_img(img_path, target_size=(img_size, img_size))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return preprocess_func(img_tensor)


def predict_board_keras(
    model_path, img_size, pre_input, path="", a1_pos="", test=False, previous_fen=None
):
    """
    Predict the fen notation of a chessboard using Keras for inference.

    :param model_path: Path to the Keras model.
    :param img_size: Model image input size.
    :param pre_input: Model preprocess input function.
    :param path: Path to the board or directory to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg' or '../predictions/'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param test: Activates the testing function.
    :param previous_fen: FEN string of the previous board position.
    :return: Predicted FEN string representing the chessboard.
    """
    model = load_model(model_path)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    if test:
        test_predict_board(obtain_pieces_probs)
    else:
        if os.path.isdir(path):
            return continuous_predictions(path, a1_pos, obtain_pieces_probs)
        else:
            return predict_board(
                path, a1_pos, obtain_pieces_probs, previous_fen=previous_fen
            )


def predict_board_onnx(
    model_path, img_size, pre_input, path="", a1_pos="", test=False, previous_fen=None
):
    """
    Predict the fen notation of a chessboard using ONNXRuntime for
    inference.

    :param model_path: Path to the ONNX model.
    :param img_size: Model image input size.
    :param pre_input: Model preprocess input function.
    :param path: Path to the board or directory to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg' or '../predictions/'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param test: Activates the testing function.
    :param previous_fen: FEN string of the previous board position.
    :return: Predicted FEN string representing the chessboard.
    """
    sess = onnxruntime.InferenceSession(model_path)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            predictions.append(
                sess.run(None, {sess.get_inputs()[0].name: piece_img})[0][0]
            )
        return predictions

    if test:
        test_predict_board(obtain_pieces_probs)
    else:
        if os.path.isdir(path):
            return continuous_predictions(path, a1_pos, obtain_pieces_probs)
        else:
            return predict_board(
                path, a1_pos, obtain_pieces_probs, previous_fen=previous_fen
            )


def predict_board_trt(
    model_path, img_size, pre_input, path="", a1_pos="", test=False, previous_fen=None
):
    """
    Predict the fen notation of a chessboard using TensorRT for
    inference.

    :param model_path: Path to the TensorRT engine with batch size 64.
    :param img_size: Model image input size.
    :param pre_input: Model preprocess input function.
    :param path: Path to the board or directory to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg' or '../predictions/'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param test: Activates the testing function.
    :param previous_fen: FEN string of the previous board position.
    :return: Predicted FEN string representing the chessboard.
    """
    if cuda is None or trt is None:
        raise ImportError("Unable to import pycuda or tensorrt")

    class __HostDeviceTuple:
        """A tuple of host and device. Clarifies code."""

        def __init__(self, _host, _device):
            self.host = _host
            self.device = _device

    def __allocate_buffers(engine):
        """Allocates all buffers required for the specified engine."""
        inputs = []
        outputs = []
        bindings = []

        for binding in engine:
            # Get binding (tensor/buffer) size
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            # Get binding (tensor/buffer) data type (numpy-equivalent)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate page-locked memory (i.e., pinned memory) buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Allocate linear piece of device memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(__HostDeviceTuple(host_mem, device_mem))
            else:
                outputs.append(__HostDeviceTuple(host_mem, device_mem))

        stream = cuda.Stream()
        return inputs, outputs, bindings, stream

    def __infer(context, bindings, inputs, outputs, stream, batch_size=64):
        """
        Infer outputs on the IExecutionContext for the specified inputs.
        """
        # Transfer input data to the GPU
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)
        # Run inference
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )
        # Transfer predictions back from the GPU
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        stream.synchronize()

        return [out.host for out in outputs]

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    # Read and deserialize the serialized ICudaEngine
    with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    inputs, outputs, bindings, stream = __allocate_buffers(engine)

    img_array = np.zeros((engine.max_batch_size, trt.volume((img_size, img_size, 3))))

    # Create an IExecutionContext (context for executing inference)
    with engine.create_execution_context() as context:

        def obtain_pieces_probs(pieces):
            # Assuming batch size == 64
            for i, piece in enumerate(pieces):
                img_array[i] = load_image(piece, img_size, pre_input).ravel()
            np.copyto(inputs[0].host, img_array.ravel())
            trt_outputs = __infer(context, bindings, inputs, outputs, stream)[-1]

            return [trt_outputs[ind : ind + 13] for ind in range(0, 13 * 64, 13)]

        if test:
            test_predict_board(obtain_pieces_probs)
        else:
            if os.path.isdir(path):
                return continuous_predictions(path, a1_pos, obtain_pieces_probs)
            else:
                return predict_board(
                    path, a1_pos, obtain_pieces_probs, previous_fen=previous_fen
                )


def predict_board(
    board_path, a1_pos, obtain_pieces_probs, board_corners=None, previous_fen=None
):
    """
    Predict the fen notation of a chessboard.

    The obtain_predictions argument allows us to predict using different
    methods (such as Keras, ONNX or TensorRT models) that may need
    additional context.

    :param board_path: Path to the board to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param obtain_pieces_probs: Function which receives a list with the
        path to each piece image in FEN notation order and returns the
        corresponding probabilities of each piece belonging to each
        class as another list.
    :param board_corners: A list of the coordinates of the four board
        corners. If it is not None, first check if the board is in the
        position given by these corners. If not, runs the full
        detection.
    :param previous_fen: FEN string of the previous board position.
        If it is not None, improves piece inference.
    :return: A pair formed by the predicted FEN string representing the
        chessboard and the coordinates of the corners of the chessboard
        in the input image.
    """
    board_corners = detect_input_board(board_path, board_corners)
    pieces = obtain_individual_pieces(board_path)
    pieces_probs = obtain_pieces_probs(pieces)
    if previous_fen is not None and not check_validity_of_fen(previous_fen):
        print(
            "Warning: the previous FEN is ignored because it is invalid for a standard physical chess set"
        )
        previous_fen = None
    predictions = infer_chess_pieces(pieces_probs, a1_pos, previous_fen)

    board = list_to_board(predictions)
    fen = board_to_fen(board)

    return fen, board_corners


def continuous_predictions(path, a1_pos, obtain_pieces_probs):
    """
    Continuously monitors path and predicts any new jpg images added to
    this directory, printing its FEN string. This function doesn't
    return.

    :param path: Path to the board or directory to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg' or '../predictions/'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param obtain_pieces_probs: Function which receives a list with the
        path to each piece image in FEN notation order and returns the
        corresponding probabilities of each piece belonging to each
        class as another list.
    """
    if not os.path.isdir(path):
        raise ValueError("The path parameter must be a directory")

    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    print("Done loading. Monitoring " + path)
    board_corners = None
    fen = None
    processed_board = False
    while True:
        for board_path in sorted(glob.glob(path + "*.jpg"), key=natural_key):
            fen, board_corners = predict_board(
                board_path, a1_pos, obtain_pieces_probs, board_corners, fen
            )
            print(fen)
            processed_board = True
            os.remove(board_path)

        if not processed_board:
            time.sleep(0.1)


def test_predict_board(obtain_predictions):
    """Tests board prediction."""
    fens, a1_squares, previous_fens = read_correct_fen(
        os.path.join("predictions", "boards_with_previous.fen")
    )

    for i in range(5):
        fen = time_predict_board(
            os.path.join("predictions", "test" + str(i + 1) + ".jpg"),
            a1_squares[i],
            obtain_predictions,
        )
        print_fen_comparison(
            "test" + str(i + 1) + ".jpg",
            fen,
            fens[i],
            False,
        )

        if previous_fens[i] is not None and not check_validity_of_fen(previous_fens[i]):
            print(
                f"Warning: the previous FEN for test{i + 1}.jpg is ignored because it is invalid for a standard physical chess set\n"
            )
            previous_fens[i] = None

        if previous_fens[i] is not None:
            fen = time_predict_board(
                os.path.join("predictions", "test" + str(i + 1) + ".jpg"),
                a1_squares[i],
                obtain_predictions,
                previous_fens[i],
            )
            print_fen_comparison("test" + str(i + 1) + ".jpg", fen, fens[i], True)


def detect_input_board(board_path, board_corners=None):
    """
    Detects the input board and stores the result as 'tmp/board_name in
    the folder containing the board. If the folder tmp exists, deletes
    its contents. If not, creates the tmp folder.

    :param board_path: Path to the board to detect. Must have rw
        permission.
        For example: '../predictions/board.jpg'.
    :param board_corners: A list of the coordinates of the four board
        corners. If it is not None, first check if the board is in the
        position given by these corners. If not, runs the full
        detection.
    :return: A list of the new coordinates of the four board corners
        detected.
    """
    input_image = cv2.imread(board_path)
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    image_object = detect(input_image, os.path.join(head, "tmp", tail), board_corners)
    board_corners, _ = compute_corners(image_object)
    return board_corners


def obtain_individual_pieces(board_path):
    """
    Obtain the individual pieces of a board.

    :param board_path: Path to the board to detect. Must have rw
        permission. The detected board should be in a tmp folder as done
        by detect_input_board.
        For example: '../predictions/board.jpg'.
    :return: List with the path to each piece image.
    """
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    pieces_dir = os.path.join(tmp_dir, "pieces/")
    os.mkdir(pieces_dir)
    split_square_board_image(os.path.join(tmp_dir, tail), "", pieces_dir)
    return sorted(glob.glob(pieces_dir + "/*.jpg"))


def time_predict_board(board_path, a1_pos, obtain_pieces_probs, previous_fen=None):
    """
    Predict the fen notation of a chessboard. Prints the elapsed times.

    The obtain_predictions argument allows us to predict using different
    methods (such as Keras, ONNX or TensorRT models) that may need
    additional context.

    :param board_path: Path to the board to detect. Must have rw permission.
        For example: '../predictions/board.jpg'.
    :param a1_pos: Position of the a1 square. Must be one of the
        following: "BL", "BR", "TL", "TR".
    :param obtain_pieces_probs: Function which receives a list with the
        path to each piece image in FEN notation order and returns the
        corresponding probabilities of each piece belonging to each
        class as another list.
    :param previous_fen: FEN string of the previous board position.
    :return: Predicted fen string representing the chessboard.
    """
    total_time = 0

    start = time.perf_counter()
    detect_input_board(board_path)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time detecting the input board: {elapsed_time}")

    start = time.perf_counter()
    pieces = obtain_individual_pieces(board_path)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time obtaining the individual pieces: {elapsed_time}")

    start = time.perf_counter()
    pieces_probs = obtain_pieces_probs(pieces)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time predicting probabilities: {elapsed_time}")

    start = time.perf_counter()
    predictions = infer_chess_pieces(pieces_probs, a1_pos, previous_fen)
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


def print_fen_comparison(board_name, fen, correct_fen, used_previous_fen):
    """
    Compares the predicted fen with the correct fen and pretty prints
    the result.

    :param board_name: Name of the board. For example: 'test1.jpg'
    :param fen: Predicted fen string.
    :param correct_fen: Correct fen string.
    :param used_previous_fen: Whether previous fen was used during prediction.
    """
    n_dif = compare_fen(fen, correct_fen)
    used_previous_fen_str = (
        "_with_previous_fen" if used_previous_fen else "_without_previous_fen"
    )
    print(
        board_name[:-4]
        + used_previous_fen_str
        + " - Err:"
        + str(n_dif)
        + " Acc:{:.2f}% FEN:".format((1 - (n_dif / 64)) * 100)
        + fen
        + "\n"
    )


def read_correct_fen(fen_file):
    """
    Reads the correct fen for testing from fen_file.

    :param board_path: Path to the file containing the correct fens,
        a1 squares, and (optionally) correct previous fens.
    :return: List with the correct fens, list with the correct
        a1 squares, and list with correct previous fens.
    """
    fens = []
    a1_squares = []
    previous_fens = []

    with open(fen_file, "r") as fen_fd:
        lines = fen_fd.read().splitlines()
        for line in lines:
            line = line.split()
            if not len(line) in [2, 3]:
                raise ValueError(
                    "All lines in fen file must have the format "
                    "'fen orientation [previous_fen]'"
                )
            fens.append(line[0])
            a1_squares.append(line[1])
            if len(line) == 2:
                previous_fens.append(None)
            else:
                previous_fens.append(line[2])
    return fens, a1_squares, previous_fens


def check_validity_of_fen(fen: str) -> bool:
    """
    Checks the validity of the FEN string (assuming a standard physical chess set).

    :param fen: FEN string whose validity is to be checked.
    :return: Whether the input FEN is valid or not.
    """
    board = chess.Board(fen)
    if not board.is_valid():  # If it's white to move, the FEN is invalid
        board.turn = chess.BLACK
        if not board.is_valid():  # If it's black to move, the FEN is also invalid
            return False

    num_of_P = fen.count("P")  # Number of white pawns
    num_of_Q = fen.count("Q")  # Number of white queens
    num_of_R = fen.count("R")  # Number of white rooks
    num_of_N = fen.count("N")  # Number of white knights
    num_of_p = fen.count("p")  # Number of black pawns
    num_of_q = fen.count("q")  # Number of black queens
    num_of_r = fen.count("r")  # Number of black rooks
    num_of_n = fen.count("n")  # Number of black knights
    fen_list = board_to_list(fen_to_board(fen))
    num_of_light_squared_B = sum(
        [
            is_white_square(square)
            for (square, piece_type) in enumerate(fen_list)
            if piece_type == "B"
        ]
    )  # Number of light-squared bishops for white
    num_of_dark_squared_B = (
        fen.count("B") - num_of_light_squared_B
    )  # Number of dark-squared bishops for white
    num_of_light_squared_b = sum(
        [
            is_white_square(square)
            for (square, piece_type) in enumerate(fen_list)
            if piece_type == "b"
        ]
    )  # Number of light-squared bishops for black
    num_of_dark_squared_b = (
        fen.count("b") - num_of_light_squared_b
    )  # Number of dark-squared bishops for black

    if (
        num_of_R > 2
        or num_of_r > 2
        or num_of_N > 2
        or num_of_n > 2
        or (num_of_light_squared_B + num_of_dark_squared_B) > 2
        or (num_of_light_squared_b + num_of_dark_squared_b) > 2
        or num_of_Q > 2
        or num_of_q > 2
    ):  # Number of some piece does not make sense for a standard physical chess set
        return False

    if (
        num_of_P == 7
        and num_of_Q == 2  # A white pawn has promoted into a queen
        and (
            num_of_light_squared_B == 2 or num_of_dark_squared_B == 2
        )  # A white pawn has promoted into a bishop
    ):
        return False

    if num_of_P == 8 and (
        num_of_Q == 2  # A white pawn has promoted into a queen
        or (num_of_light_squared_B == 2 or num_of_dark_squared_B == 2)
    ):  # A white pawn has promoted into a bishop
        return False

    if (
        num_of_p == 7
        and num_of_q == 2  # A black pawn has promoted into a queen
        and (
            num_of_light_squared_b == 2 or num_of_dark_squared_b == 2
        )  # A black pawn has promoted into a bishop
    ):
        return False

    if num_of_p == 8 and (
        num_of_q == 2  # A black pawn has promoted into a queen
        or (num_of_light_squared_b == 2 or num_of_dark_squared_b == 2)
    ):  # A black pawn has promoted into a bishop
        return False

    return True
