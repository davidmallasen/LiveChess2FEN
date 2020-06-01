"""
Executes the detection of a chessboard.
"""
from lc2fen.board2data import regenerate_data_state, process_input_boards


def main():
    regenerate_data_state("data")
    process_input_boards("data")


if __name__ == "__main__":
    main()
