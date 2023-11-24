# LiveChess2FEN

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LiveChess2FEN is a fully functional framework that automatically digitizes the
configuration of a chessboard. It is optimized for execution on an Nvidia
Jetson Nano (but it also works on any computer).

This repository contains the code used in our
[paper](https://arxiv.org/abs/2012.06858). If you find this useful, please
consider citing us.

~~~bibtex
@article{mallasen2020LiveChess2FEN,
  title = {LiveChess2FEN: A Framework for Classifying Chess Pieces Based on CNNs},
  author = {Mallas{\'e}n Quintana, David and Del Barrio Garc{\'i}a, Alberto Antonio and Prieto Mat{\'i}as, Manuel},
  year = {2020},
  month = dec,
  journal = {arXiv:2012.06858 [cs]},
  eprint = {2012.06858},
  eprinttype = {arxiv},
  url = {http://arxiv.org/abs/2012.06858},
  archiveprefix = {arXiv}
}
~~~

![Digitization process](docs/complete_method.png)

## Benchmarks

The following testing data have been obtained with the Nvidia Jetson
Nano 4GB. Each time value represents how long it takes to perform an
operation on a single chessboard.

### Piece-classification times

![Runtime vs accuracy Pareto front](docs/runtime_vs_accuracy_wfront.png)

![Summary of time and accuracy](docs/piece_classification_times.png)

### Full-digitization times

![Full digitization times](docs/full_digitization_times_summary.png)

### Static-digitization times

_See `lc2fen/detectboard/laps.py -> check_board_position()`_

![Static board digitization times](docs/static_digitization_times_summary.png)

## Installation instructions

Follow the installation instructions for your specific computer.
After this, you will be ready to use LiveChess2FEN by following
the [usage instructions](#usage-instructions). Note that you will
need at least Python 3.9 installed in your system.

<details><summary>Jetson Nano</summary><p>

Instructions for JetPack 4.6 are presented below. If you run into any problems,
see the [Troubleshooting](#troubleshooting) section. You can find a list of the python packages required in the `requirements.txt` file.

1. Install [tensorflow for Jetson Nano](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770):

    ~~~bash
    sudo apt-get update
    sudo apt-get install -y python3-pip pkg-config
    sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
    sudo pip3 install --verbose 'protobuf<4' 'Cython<3'
    sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
    sudo pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
    ~~~

2. Install ONNX Runtime

    Download the .whl file from [here](https://nvidia.box.com/s/bfs688apyvor4eo8sf3y1oqtnarwafww) and run

    ~~~bash
    sudo pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
    ~~~

3. Install OpenCV 4.5 with CUDA enabled. To do so, download and execute
[this script](https://github.com/AastaNV/JEP/blob/b5209e3edfad0f3f6b33e0cbc7e15ca3a49701cf/script/install_opencv4.5.0_Jetson.sh). Warning: this process will take some time and
you may need to increase the swap space with `jtop`.

4. If you plan on [converting ONNX models to TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#convert-onnx-engine), add the following lines
to the `.bashrc` file to access `trtexec`:

    ~~~bash
    export PATH=$PATH:/usr/src/tensorrt/bin
    ~~~

    Then, you can run `trtexec --onnx=model.onnx --saveEngine=model.trt` to convert an ONNX model to a TensorRT engine.

5. Install the rest of the required packages:

    ~~~bash
    sudo pip3 install -r requirements.txt
    ~~~

### Utilities

- You can also install [jtop](https://github.com/rbonghi/jetson_stats) to
monitor the usage of the Jetson Nano. To install, run

    ~~~bash
    sudo pip3 install -U jetson-stats
    ~~~

    and reboot the Jetson Nano. You can execute it by running `jtop`.

### Troubleshooting

- To upgrade CMake, download
[CMake 3.14.7](https://cmake.org/files/v3.14/cmake-3.14.7.tar.gz) and run

    ~~~bash
    tar -zxvf cmake-3.14.7.tar.gz
    cd cmake-3.14.7
    sudo apt-get install libcurl4-openssl-dev
    sudo ./bootstrap
    sudo make
    sudo make install
    cmake --version
    ~~~

- To install [protobuf](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) download [protobuf 3.17.3](https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protobuf-cpp-3.17.3.tar.gz) and run

    ~~~bash
    tar -zxvf protobuf-cpp-3.17.3.tar.gz
    cd protobuf-3.17.3
    ./configure
    make
    sudo make install
    sudo ldconfig
    ~~~

- If you get the error `ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block`, run

    ~~~bash
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    ~~~

    In order to permanently fix the error, add that line to the end of your
    `~/.bashrc` file.

- If you get the error `Illegal instruction (core dumped)`, run

    ~~~bash
    export OPENBLAS_CORETYPE=ARMV8
    ~~~

    In order to permanently fix the error, add that line to the end of your
    `~/.bashrc` file.

- If you get the
error `error: command 'aarch64-linux-gnu-gcc' failed with exit status 1`, run

    ~~~bash
    sudo apt-get install python3-dev
    ~~~

- If you cannot install `pycuda` because it doesn't find `cuda.h`, run

    ~~~bash
    export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib
    ~~~

    In order to permanently fix the error, add those lines to the end of your
    `~/.bashrc` file.

- In any case, if you find that there is a library missing, you can try to install
it using pip or google how to install it on the Jetson Nano.

</p></details>

<details><summary>Ubuntu PC</summary><p>

Installation instructions for Ubuntu (22.04) are presented below. Other Linux distributions should be similar.

1. First clone the repository and `cd` into it:

    ~~~bash
    git clone https://github.com/davidmallasen/LiveChess2FEN.git
    cd LiveChess2FEN
    ~~~

2. Create a python virtual environment, activate it and upgrade pip:

    ~~~bash
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ~~~

    You will have to activate the virtual environment every time you want to use LiveChess2FEN.

3. Install the required python packages:

    ~~~bash
    pip install -r requirements_pc.txt
    ~~~

    This should include the following packages:
    - NumPy
    - OpenCV4
    - Matplotlib
    - Scikit-learn
    - Pillow
    - Pyclipper
    - Tqdm

    Also, depending on the inference engine you want to use, install the following dependencies:
    - Keras with tensorflow backend.
    - ONNX Runtime.
    - (Optional) TensorRT and PyCUDA.

</p></details>

<details><summary>Windows PC</summary><p>

Installation instructions for a Windows computer are presented below.

1. First, install Python 3.10 from Microsoft Store. It is important NOT to
install Python 3.11 instead as
it would create dependency (`numpy`) issues when we later install `onnxruntime` and `tensorflow==2.12.0`.

2. Then make sure your pip is up to date by running the following command in
Windows PowerShell:

    ~~~bash
    pip install --upgrade pip
    ~~~

3. If you see any warning about some directory not on PATH, follow [this](https://stackoverflow.com/questions/49966547/pip-10-0-1-warning-consider-adding-this-directory-to-path-or/51165784#51165784)
and restart the computer to resolve it.

4. In order to successfully install `tensorflow`, you need to first [enable
long paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later). To do
so, open another PowerShell as administrator and run the following command:

    ~~~text
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ~~~

5. Now you can install all the relevant packages by running the following
commands in Windows PowerShell:

    ~~~bash
    pip install numpy
    pip install opencv-python
    pip install chess
    pip install tensorflow==2.12.0
    pip install onnxruntime
    pip install matplotlib
    pip install pyclipper
    pip install scikit-learn
    pip install tqdm
    pip install pandas
    pip install onnx
    pip install tf2onnx
    pip install pytest
    ~~~

    Note: the above commands would install all the latest-possible versions of
    the required packages (it was found that there might not be any
    restrictions on the versions of non`tensorflow` packages). Alternatively, you could use the
    "requirements_pc.txt" file (`pip install -r requirements_pc.txt`) to
    install the specific versions that have been
    tested to be 100% working.

6. Finally, in order to successfully import `tensorflow`, you also need to
install a Microsoft Visual C++ Redistributable package from
[here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
If you are using Windows 11 ([Windows 11 only has the 64-bit version](https://www.intowindows.com/where-can-i-download-windows-11-32-bit-iso/)), you can simply download and install
[this](https://aka.ms/vs/17/release/vc_redist.x64.exe).

</p></details>

## Usage instructions

1. Download the prediction models (the `.h5`, `.onnx` or `.trt` files) from the
 [releases](https://github.com/davidmallasen/LiveChess2FEN/releases)
 and save them to the `data/models` folder.

2. Download the contents of `TestImages.zip->FullDetection` from the
[releases](https://github.com/davidmallasen/LiveChess2FEN/releases) into the
`data/predictions` folder. You should have 5 test images and 2 `.fen` files.

3. Edit `test_lc2fen.py` and set the `ACTIVATE_*`, `MODEL_PATH_*`,
 `IMG_SIZE_*`, and `PRE_INPUT_*` constants.

   - `ACTIVATE_KERAS = True` will select Keras with tensorflow backend as the
   inference engine. The Keras engine is the slowest of the three.

   - `ACTIVATE_ONNX = True` will select ONNX Runtime as the inference engine.
   It is significantly faster than Keras but almost just as accurate. It is the
   recommended choice for any standard computer.

   - `ACTIVATE_TRT = True` will select TensorRT as the inference engine. It is
   the fastest of the three but only available on computers with Nvidia GPUs.

4. Run the `test_lc2fen.py` script.

5. You can then use LiveChess2FEN by repeating steps 3 and 4 with the
`lc2fen.py` program instead of the `test_lc2fen.py` script. Run
`python3 lc2fen.py -h` to display the help message.

## Training new models

To train new models, check the `cpmodels` folder. That directory contains 
the python scripts used to train the chess piece models. It also contains 
the scripts used to manipulate both the dataset and the models.

### Setup

1. Download the dataset from the [releases](https://github.com/davidmallasen/LiveChess2FEN/releases)
into the `data/dataset` directory.
2. Unzip the dataset into the same directory. It should be in `data/dataset/ChessPieceModels/`.
3. Use the functions in `cpmodels/dataset.py` to split the dataset into
training and validation sets. The training set should be in `data/dataset/train/`
and the validation set should be in `data/dataset/validation/`. You can
do this by running `python` and then:

    ~~~python
    from cpmodels.dataset import split_dataset
    split_dataset()
    exit()
    ~~~

4. Use the `train_*.py` scripts to train the models. The models will be
saved in `cpmodels/models/`.
5. When you want to use a trained model with LiveChess2FEN, copy the
model file into `data/models/`.

## Testing

LiveChess2FEN supports [pytest](https://docs.pytest.org/en/latest/) unit
testing. All tests are located in the `test` folder. To run the tests,
simply run:

~~~bash
pytest -rA -v
~~~

## Contributing

Contributions are very welcome! Please check the [CONTRIBUTING](CONTRIBUTING.md)
file for more information on how to contribute to LiveChess2FEN.

## License

You can find a non-legal quick summary here: [tldrlegal AGPL](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0))

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
