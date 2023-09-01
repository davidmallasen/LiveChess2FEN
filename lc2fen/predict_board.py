"""This module is responsible for predicting board configurations."""


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

    # `pycuda.autoinit` enables pycuda to automatically manage CUDA
    # context creation and cleanup.
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
    is_light_square,
    fen_to_board,
    board_to_list,
)
from lc2fen.infer_pieces import infer_chess_pieces
from lc2fen.split_board import split_board_image_trivial


def load_image(img_path: str, img_size: int, preprocess_func) -> np.ndarray:
    """Load an image.

    This function loads an image from its path. It is intended to be
    used for loading piece images.

    :param img_path: Image path.

    :param img_size: Size of the input image. Example: `224`.

    :param preprocess_func: Preprocessing fuction for the input image.

    :return: Preprocessed image.
    """
    img = load_img(img_path, target_size=(img_size, img_size))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return preprocess_func(img_tensor)


def predict_board_keras(
    model_path: str,
    img_size: int,
    pre_input,
    path="",
    a1_pos="",
    test=False,
    previous_fen: (str | None) = None,
) -> tuple[str, int] | None:
    """Predict FEN(s) from board image(s) using Keras for inference.

    This function predicts FEN string(s) from chessboard images using
    Keras as the inference engine.

    :param model_path: Path to the Keras model (ending with ".h5").

    :param img_size: Input size for the model.

    :param pre_input: Input-preprocessing function for the model.

    :param path: Path to the chessboard image or folder.

        This is the path to either a single chessboard image or a folder
        that contains chessboard images.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"` or `"../predictions/"`.

    :param a1_pos: Position of the a1 square of the chessboard images.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image(s).

    :param test: Whether to activate testing mode.

        If `test` is `True`, `path` is not used.

    :param previous_fen: FEN string of the previous board position.

        This parameter is only used when `path` points to a single image
        and `test` is `False`.

    :return: Predicted FEN string of the current board position for
    single-FEN prediction.

        If `test` is `True`, the function returns `None`.

        If `path` points to a folder, the function does not return.
    """
    model = load_model(model_path)

    def obtain_piece_probs_for_all_64_squares(
        pieces: list[str],
    ) -> list[list[float]]:
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    if test:
        test_predict_board(obtain_piece_probs_for_all_64_squares)
    else:
        if os.path.isdir(path):
            return continuous_predictions(
                path, a1_pos, obtain_piece_probs_for_all_64_squares
            )
        else:
            return predict_board(
                path,
                a1_pos,
                obtain_piece_probs_for_all_64_squares,
                previous_fen=previous_fen,
            )


def predict_board_onnx(
    model_path: str,
    img_size: int,
    pre_input,
    path="",
    a1_pos="",
    test=False,
    previous_fen: (str | None) = None,
) -> tuple[str, int] | None:
    """Predict FEN(s) from board image(s) using ONNX for inference.

    This function predicts FEN string(s) from chessboard image(s) using
    ONNXRuntime as the inference engine.

    :param model_path: Path to the ONNX model (ending with ".onnx").

    :param img_size: Input size for the model.

    :param pre_input: Input-preprocessing function for the model.

    :param path: Path to the chessboard image or folder.

        This is the path to either a single chessboard image or a folder
        that contains chessboard images.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"` or `"../predictions/"`.

    :param a1_pos: Position of the a1 square of the chessboard images.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image(s).

    :param test: Whether to activate testing mode.

        If `test` is `True`, `path` is not used.

    :param previous_fen: FEN string of the previous board position.

        This parameter is only used when `path` points to a single image
        and `test` is `False`.

    :return: Predicted FEN string of the current board position for
    single-FEN prediction.

        If `test` is `True`, the function returns `None`.

        If `path` points to a folder, the function does not return.
    """
    sess = onnxruntime.InferenceSession(model_path)

    def obtain_piece_probs_for_all_64_squares(
        pieces: list[str],
    ) -> list[list[float]]:
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            predictions.append(
                sess.run(None, {sess.get_inputs()[0].name: piece_img})[0][0]
            )
        return predictions

    if test:
        test_predict_board(obtain_piece_probs_for_all_64_squares)
    else:
        if os.path.isdir(path):
            return continuous_predictions(
                path, a1_pos, obtain_piece_probs_for_all_64_squares
            )
        else:
            return predict_board(
                path,
                a1_pos,
                obtain_piece_probs_for_all_64_squares,
                previous_fen=previous_fen,
            )


def predict_board_trt(
    model_path: str,
    img_size: int,
    pre_input,
    path="",
    a1_pos="",
    test=False,
    previous_fen: (str | None) = None,
) -> tuple[str, int] | None:
    """Predict FEN(s) from board image(s) using TensorRT for inference.

    This function predicts FEN string(s) from chessboard image(s) using
    TensorRT as the inference engine.

    :param model_path: Path to the TensorRT engine with batch size 64.

        The path must end with the ".trt" extension.

    :param img_size: Input size for the model.

    :param pre_input: Input-preprocessing function for the model.

    :param path: Path to the chessboard image or folder.

        This is the path to either a single chessboard image or a folder
        that contains chessboard images.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"` or `"../predictions/"`.

    :param a1_pos: Position of the a1 square of the chessboard images.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image(s).

    :param test: Whether to activate testing mode.

        If `test` is `True`, `path` is not used.

    :param previous_fen: FEN string of the previous board position.

        This parameter is only used when `path` points to a single image
        and `test` is `False`.

    :return: Predicted FEN string of the current board position for
    single-FEN prediction.

        If `test` is `True`, the function returns `None`.

        If `path` points to a folder, the function does not return.
    """
    if cuda is None or trt is None:
        raise ImportError("Unable to import pycuda or tensorrt")

    class __HostDeviceTuple:
        """A tuple of host and device. It helps clarify code."""

        def __init__(self, _host, _device):
            self.host = _host
            self.device = _device

    def __allocate_buffers(engine):
        """Allocate all buffers required for the specified engine."""
        inputs = []
        outputs = []
        bindings = []

        for binding in engine:
            # Get binding (tensor/buffer) size
            size = (
                trt.volume(engine.get_binding_shape(binding))
                * engine.max_batch_size
            )
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
        """Infer outputs on IExecutionContext for specified inputs."""
        # Transfer input data to the GPU
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)
        # Run inference
        context.execute_async(
            batch_size=batch_size,
            bindings=bindings,
            stream_handle=stream.handle,
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

    img_array = np.zeros(
        (engine.max_batch_size, trt.volume((img_size, img_size, 3)))
    )

    # Create an IExecutionContext (context for executing inference)
    with engine.create_execution_context() as context:

        def obtain_piece_probs_for_all_64_squares(
            pieces: list[str],
        ) -> list[list[float]]:
            # Assuming batch size == 64
            for i, piece in enumerate(pieces):
                img_array[i] = load_image(piece, img_size, pre_input).ravel()
            np.copyto(inputs[0].host, img_array.ravel())
            trt_outputs = __infer(context, bindings, inputs, outputs, stream)[
                -1
            ]

            return [
                trt_outputs[ind : ind + 13] for ind in range(0, 13 * 64, 13)
            ]

        if test:
            test_predict_board(obtain_piece_probs_for_all_64_squares)
        else:
            if os.path.isdir(path):
                return continuous_predictions(
                    path, a1_pos, obtain_piece_probs_for_all_64_squares
                )
            else:
                return predict_board(
                    path,
                    a1_pos,
                    obtain_piece_probs_for_all_64_squares,
                    previous_fen=previous_fen,
                )


def predict_board(
    board_path: str,
    a1_pos: str,
    obtain_piece_probs_for_all_64_squares,
    board_corners: (list[list[int]] | None) = None,
    previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]]:
    """Predict the FEN string from a chessboard image.

    :param board_path: Path to the chessboard image of interest.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"`.

    :param a1_pos: Position of the a1 square of the chessboard image.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image.

    :param obtain_piece_probs_for_all_64_squares: Path-to-prob function.

        This function takes as input a length-64 list of paths to
        chess-piece images and returns a length-64 list of the
        corresponding piece probabilities (each element of the list is a
        length-13 sublist that contains 13 piece probabilities).

        This parameter allows us to deploy different inference engines
        (Keras, ONNX, or TensorRT).

    :param board_corners: Length-4 list of coordinates of four corners.

        The 4 board corners are in the order of top left, top right,
        bottom right, and bottom left.

        If it is not `None` and the corner coordinates are accurate
        enough, the neural-network-based board-detection step is skipped
        (which means the total processing time is reduced).

    :param previous_fen: FEN string of the previous board position.

        If it is not `None`, it could significantly improve the accuracy
        of FEN prediction.

    :return: A pair formed by the predicted FEN string and the
    coordinates of the corners of the chessboard in the input image.
    """
    board_corners = detect_input_board(board_path, board_corners)
    pieces = obtain_individual_pieces(board_path)
    probs_with_no_indices = obtain_piece_probs_for_all_64_squares(pieces)
    if previous_fen is not None and not check_validity_of_fen(previous_fen):
        print(
            "Warning: the previous FEN is ignored because it is invalid for a "
            "standard physical chess set"
        )
        previous_fen = None
    predictions = infer_chess_pieces(
        probs_with_no_indices, a1_pos, previous_fen
    )

    board = list_to_board(predictions)
    fen = board_to_fen(board)

    return fen, board_corners


def continuous_predictions(
    path: str, a1_pos: str, obtain_piece_probs_for_all_64_squares
):
    """Predict FEN strings from chessboard images continuously.

    This function continuously monitors a folder and predicts the FEN
    strings for new jpg images added to the folder. The FEN string is
    printed out every time a prediction is completed. Note that this
    function does not return.

    :param path: Path to the folder that contains chessboard image(s).

        Example: '../predictions/'.

    :param a1_pos: Position of the a1 square of the chessboard image(s).

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image(s).

    :param obtain_piece_probs_for_all_64_squares: Path-to-prob function.

        This function takes as input a length-64 list of paths to
        chess-piece images and returns a length-64 list of the
        corresponding piece probabilities (each element of the list is a
        length-13 sublist that contains 13 piece probabilities).
    """
    if not os.path.isdir(path):
        raise ValueError("The input path must point to a folder")

    def natural_key(text):
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

    print("Done loading. Monitoring " + path)
    board_corners = None
    fen = None
    processed_board = False
    while True:
        for board_path in sorted(glob.glob(path + "*.jpg"), key=natural_key):
            fen, board_corners = predict_board(
                board_path,
                a1_pos,
                obtain_piece_probs_for_all_64_squares,
                board_corners,
                fen,
            )
            print(fen)
            processed_board = True
            os.remove(board_path)

        if not processed_board:
            time.sleep(0.1)


def test_predict_board(obtain_piece_probs_for_all_64_squares):
    """Test `predict_board()`.

    :param obtain_piece_probs_for_all_64_squares: Path-to-prob function.

        This function takes as input a length-64 list of paths to
        chess-piece images and returns a length-64 list of the
        corresponding piece probabilities (each element of the list is a
        length-13 sublist that contains 13 piece probabilities).
    """
    fens, a1_squares, previous_fens = read_correct_fen(
        os.path.join("predictions", "boards_with_previous.fen")
    )

    for i in range(5):
        fen = time_predict_board(
            os.path.join("predictions", "test" + str(i + 1) + ".jpg"),
            a1_squares[i],
            obtain_piece_probs_for_all_64_squares,
        )
        print_fen_comparison(
            "test" + str(i + 1) + ".jpg",
            fen,
            fens[i],
            False,
        )

        if previous_fens[i] is not None and not check_validity_of_fen(
            previous_fens[i]
        ):
            print(
                f"Warning: the previous FEN for test{i + 1}.jpg is ignored "
                "because it is invalid for a standard physical chess set\n"
            )
            previous_fens[i] = None

        if previous_fens[i] is not None:
            fen = time_predict_board(
                os.path.join("predictions", "test" + str(i + 1) + ".jpg"),
                a1_squares[i],
                obtain_piece_probs_for_all_64_squares,
                previous_fens[i],
            )
            print_fen_comparison(
                "test" + str(i + 1) + ".jpg", fen, fens[i], True
            )


def detect_input_board(
    board_path: str, board_corners: (list[list[int]] | None) = None
) -> list[list[int]]:
    """Detect the input board.

    This function takes as input a path to a chessboard image
    (e.g., "image.jpg") and stores the image that contains the detected
    chessboard in the "tmp" subfolder of the folder containing the board
    (e.g., "tmp/image.jpg").

    If the "tmp" folder already exists, the function deletes its
    contents. Otherwise, the function creates the "tmp" folder.

    :param board_path: Path to the chessboard image of interest.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"`.

    :param board_corners: Length-4 list of coordinates of four corners.

        The 4 board corners are in the order of top left, top right,
        bottom right, and bottom left.

        If it is not `None` and the corner coordinates are accurate
        enough, the neural-network-based board-detection step is skipped
        (which means the total processing time is reduced).

    :return: Length-4 list of the (new) coordinates of the four board
    corners detected.
    """
    input_image = cv2.imread(board_path)
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    image_object = detect(
        input_image, os.path.join(head, "tmp", tail), board_corners
    )
    board_corners, _ = compute_corners(image_object)
    return board_corners


def obtain_individual_pieces(board_path: str) -> list[str]:
    """Obtain the individual pieces of a board.

    :param board_path: Path to the chessboard image of interest.

        The path must have rw permission.

        The image of the detected chessboard must be in the
        corresponding "tmp" folder (see `detect_input_board()`).

        Example: `"../predictions/board.jpg"`.

    :return: Length-64 list of paths to chess-piece images
    """
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    pieces_dir = os.path.join(tmp_dir, "pieces/")
    os.mkdir(pieces_dir)
    split_board_image_trivial(os.path.join(tmp_dir, tail), "", pieces_dir)
    return sorted(glob.glob(pieces_dir + "/*.jpg"))


def time_predict_board(
    board_path,
    a1_pos,
    obtain_piece_probs_for_all_64_squares,
    previous_fen=None,
):
    """Time the FEN-prediction process.

    This function predicts the FEN string from a chessboard and prints
    out the elapsed times during the prediction.

    :param board_path: Path to the chessboard image of interest.

        The path must have rw permission.

        Example: `"../predictions/board.jpg"`.

    :param a1_pos: Position of the a1 square of the chessboard image.

        This is the position of the a1 square (`"BL"`, `"BR"`, `"TL"`,
        or `"TR"`) corresponding to the chessboard image.

    :param obtain_piece_probs_for_all_64_squares: Path-to-prob function.

        This function takes as input a length-64 list of paths to
        chess-piece images and returns a length-64 list of the
        corresponding piece probabilities (each element of the list is a
        length-13 sublist that contains 13 piece probabilities).

        This parameter allows us to deploy different inference engines
        (Keras, ONNX, or TensorRT).

    :param previous_fen: FEN string of the previous board position.

    :return: Predicted FEN string corresponding to the input chessboard
    image.
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
    probs_with_no_indices = obtain_piece_probs_for_all_64_squares(pieces)
    elapsed_time = time.perf_counter() - start
    total_time += elapsed_time
    print(f"Elapsed time predicting probabilities: {elapsed_time}")

    start = time.perf_counter()
    predictions = infer_chess_pieces(
        probs_with_no_indices, a1_pos, previous_fen
    )
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


def print_fen_comparison(
    board_name: str, fen: str, correct_fen: str, used_previous_fen: bool
):
    """Compare predicted FEN with correct FEN and pretty-print result.

    :param board_name: Filename of the chessboard image.

        Example: `"test1.jpg"`.

    :param fen: Predicted FEN string.

    :param correct_fen: Correct FEN string.

    :param used_previous_fen: Whether the FEN string of the previous
    board position was used during the prediction.
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


def read_correct_fen(
    fen_file: str,
) -> tuple[list[str], list[str], list[str | None]]:
    """Read the correct FENs.

    :param fen_file: Path to the correct-FEN file.

        This files contains the correct FENs, a1-square positions, and
        (optionally) correct previous FENs.

    :return: Length-3 tuple of the correct-FEN information.

        The first element of the tuple is a list of the correct FENs,
        the second is a list of the corresponding a1-square positions,
        and the third is a list of the corresponding correct previous
        FENs.

        Any `None` in the third list represents an unknown previous
        board position.
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
    """Check validity of FEN assuming a standard physical chess set.

    This function checks the validity of a FEN string assuming a
    standard physical chess set.

    :param fen: FEN string whose validity is to be checked.

    :return: Whether the input FEN string is valid or not.
    """
    board = chess.Board(fen)
    if not board.is_valid():  # If it's white to move, the FEN is invalid
        board.turn = chess.BLACK
        if (
            not board.is_valid()
        ):  # If it's black to move, the FEN is also invalid
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
            is_light_square(square)
            for (square, piece_type) in enumerate(fen_list)
            if piece_type == "B"
        ]
    )  # Number of light-squared bishops for white
    num_of_dark_squared_B = (
        fen.count("B") - num_of_light_squared_B
    )  # Number of dark-squared bishops for white
    num_of_light_squared_b = sum(
        [
            is_light_square(square)
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
    ):  # Number of any piece is too large for a standard physical chess set
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
