# Chess Piece Models

This directory contains the python scripts used to train the chess piece
models. It also contains the scripts used to manipulate both the dataset
and the models.

## Setup

1. Download the dataset from the [releases](https://github.com/davidmallasen/LiveChess2FEN/releases)
into the `data/dataset` directory in the root of LiveChess2FEN.
2. Unzip the dataset into the same directory. It should be in `data/dataset/ChessPieceModels/`.
3. Use the functions in `dataset.py` to randomize the dataset and split
it into training and validation sets. The training set should be in
`data/dataset/train/` and the validation set should be in `data/dataset/validation/`.
4. Use the `train_*.py` scripts to train the models. The models will be
saved in `cpmodels/models/`.
5. When you want to use a trained model with LiveChess2FEN, copy the
model file into `data/models/`.
