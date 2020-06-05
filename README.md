# LiveChess2FEN

LiveChess2FEN is a fully functional framework that automatically digitizes
the configuration of a chessboard. It is optimized for execution on a
Nvidia Jetson Nano.

![](docs/complete_method.png)

## Setup

1. Install Python 3.5 or later and the following dependencies:
    - NumPy
    - OpenCV4
    - Matplotlib
    - scikit-learn
    - pillow
    - pyclipper
    - tqdm

2. Depending on the inference engine install the following dependencies:
    - Keras with tensorflow backend
    - Onnxruntime
    - TensorRT
    
3. Create a `selected_models` folder in the project root.
4. Download the prediction models from the 
 [releases](https://github.com/davidmallasen/LiveChess2FEN/releases)
 and save them to the `selected_models` folder.
5. Create a `predictions/input_board` folder and a `predictions/pieces`
 folder in the project root.
6. Download the contents of `TestImages.zip->FullDetection` from the
[releases](https://github.com/davidmallasen/LiveChess2FEN/releases) into
 the `predictions/input_board` folder. You should have 5 test images and a
 boards.fen file.
7. Edit `lc2fen_predict.py` and set the `ACTIVATE_*`, `MODEL_PATH_*`,
 `IMG_SIZE_*` and `PRE_INPUT_*` constants.
8. Run the `lc2fen_predict.py` script.

## Contributing

Contributions are very welcome! Please check the 
[CONTRIBUTING](CONTRIBUTING.md) file for more information on how to
 contribute to LiveChess2FEN.

## License

Copyright (c) 2020 David Mallasén Quintana

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
