"""This script executes the detection of chessboards."""


from lc2fen.board2data import regenerate_data_folder, process_input_boards


def main():
    """Detect all the chessboards in the "data" folder."""
    regenerate_data_folder("data")
    process_input_boards("data")


if __name__ == "__main__":
    main()
