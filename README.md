<h1 align="center">LiveChess2FEN</h1>

<p align="center">
<a href="https://github.com/psf/black">
<img alt="Code style: black" 
src="https://img.shields.io/badge/code%20style-black-000000.svg">
</a>
</p>

LiveChess2FEN is a fully functional framework that automatically digitizes the
configuration of a chessboard. It is optimized for execution on an Nvidia
Jetson Nano (but it also works on any computer).

This repository contains the code used in our
[paper](https://arxiv.org/abs/2012.06858). If you find this useful, please
consider citing us.

~~~
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

![](docs/complete_method.png)

## Benchmarks

The following testing data have been obtained with the Nvidia Jetson
Nano 4GB. Each time value represents how long it takes to perform an
operation on a single chessboard.

#### Piece-classification times

![](docs/runtime_vs_accuracy_wfront.png)

![](docs/piece_classification_times.png)

#### Full-digitization times

![](docs/full_digitization_times_summary.png)

#### Static-digitization times

_See `lc2fen/detectboard/laps.py -> check_board_position()`_

![](docs/static_digitization_times_summary.png)

## Installation instructions

Follow the installation instructions for your specific computer.
After this, you will be ready to use LiveChess2FEN by following 
the [usage instructions](#usage-instructions).

<details><summary>Jetson Nano 2GB</summary><p>

Instructions for JetPack 4.6 are presented below. If you run into any problems,
see the [Troubleshooting](#troubleshooting) section. You can find a list of the python packages required in the `requirements.txt` file.

1. From the [Jetson Zoo](https://elinux.org/Jetson_Zoo), install:

    1. Tensorflow

        ~~~
        sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
        sudo apt-get install python3-pip
        sudo pip3 install -U pip testresources setuptools==49.6.0
        sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
        sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
        ~~~
        
    2. ONNX Runtime

        Download the .whl file from [here](https://nvidia.box.com/s/bfs688apyvor4eo8sf3y1oqtnarwafww) and run

        ~~~
        pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
        ~~~

2. Install OpenCV 4.5 with CUDA enabled. To do so, download and execute
[this script](https://github.com/AastaNV/JEP/blob/b5209e3edfad0f3f6b33e0cbc7e15ca3a49701cf/script/install_opencv4.5.0_Jetson.sh). Warning: this process will take a few hours and
you will need at least 4GB of swap space.

3. Install [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/) with the
following commands:

    ~~~
    git clone --recursive https://github.com/onnx/onnx-tensorrt.git
    cd onnx-tensorrt
    git checkout 8.0-GA
    mkdir build && cd build
    cmake .. -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTENSORRT_ROOT=/usr/src/tensorrt -DGPU_ARCHS="53"
    make
    sudo make install
    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
    ~~~

4. Install Python 3.10 (note that [Python 3.11 currently does not support ONNX
Runtime](https://github.com/microsoft/onnxruntime/issues/13482)) and the
following dependencies:

   - `numpy`
   - `chess`
   - `matplotlib`
   - `pyclipper`
   - `scikit-learn`
   - `tqdm`
   - `pandas`

    (Note: you can find a list of version numbers for the Python packages that
    have been tested to work in the `requirements.txt` file.)

### Utilities

- You can also install [jtop](https://github.com/rbonghi/jetson_stats) to
monitor the usage of the Jetson Nano. To install, run

    ~~~
    sudo -H pip install -U jetson-stats
    ~~~

    and reboot the Jetson Nano. You can execute it by running `jtop`.

### Troubleshooting

- To upgrade CMake, download
[CMake 3.14.7](https://cmake.org/files/v3.14/cmake-3.14.7.tar.gz) and run
    
    ~~~
    tar -zxvf cmake-3.14.7.tar.gz
    cd cmake-3.14.7
    sudo apt-get install libcurl4-openssl-dev
    sudo ./bootstrap
    sudo make
    sudo make install
    cmake --version
    ~~~

- To install [protobuf](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) download [protobuf 3.17.3](https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protobuf-cpp-3.17.3.tar.gz) and run
    ~~~
    tar -zxvf protobuf-cpp-3.17.3.tar.gz
    cd protobuf-3.17.3
    ./configure
    make
    sudo make install
    sudo ldconfig
    ~~~

- If you get the error `ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block`, run

    ~~~
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    ~~~

    In order to permanently fix the error, add that line to the end of your
    `~/.bashrc` file.

- If you get the error `Illegal instruction (core dumped)`, run

    ~~~
    export OPENBLAS_CORETYPE=ARMV8
    ~~~

    In order to permanently fix the error, add that line to the end of your
    `~/.bashrc` file.

- If you get the
error `error: command 'aarch64-linux-gnu-gcc' failed with exit status 1`, run

    ~~~
    sudo apt-get install python3-dev
    ~~~

</p></details>

<details><summary>Ubuntu PC</summary><p>

Installation instructions for Ubuntu (22.04) are presented below. Other Linux distributions should be similar.

1. First clone the repository and `cd` into it:
    ~~~
    git clone https://github.com/davidmallasen/LiveChess2FEN.git
    cd LiveChess2FEN
    ~~~

2. Create a python virtual environment, activate it and upgrade pip:
    ~~~
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ~~~
    You will have to activate the virtual environment every time you want to use LiveChess2FEN.

3. Install the required python packages:
    ~~~
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
[it is currently incompatible with `onnxruntime`](https://github.com/microsoft/onnxruntime/issues/13482).

2. Then make sure your pip is up to date by running the following command in
Windows PowerShell:

    `pip install --upgrade pip`

3. If you see any warning about some directory not on PATH, follow [this](https://stackoverflow.com/questions/49966547/pip-10-0-1-warning-consider-adding-this-directory-to-path-or/51165784#51165784)
and restart the computer to resolve it.

4. In order to successfully install `tensorflow`, you need to first [enable
long paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later). To do
so, open another PowerShell as administrator and run the following command:

    `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`

5. Now you can install all the relevant packages by running the following
commands in Windows PowerShell:

    ```
    pip install numpy
    pip install opencv-python
    pip install chess
    pip install tensorflow==2.11.0
    pip install onnxruntime
    pip install matplotlib
    pip install pyclipper
    pip install scikit-learn
    pip install tqdm
    pip install pandas
    pip install onnx==1.12.0
    pip install tf2onnx==1.13.0
    ```

    Note: the above commands would install all the latest-possible versions of
    the required packages (it was found that there might not be any
    restrictions on the versions of non`tensorflow`, non`onnx`, and
    non`tf2onnx` packages). Alternatively, you could use the
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

1. Create a "selected_models" folder and a "predictions" folder in the project
root.

2. Download the prediction models from the 
 [releases](https://github.com/davidmallasen/LiveChess2FEN/releases)
 and save them to the "selected_models" folder.
 
3. Download the contents of `TestImages.zip->FullDetection` from the
[releases](https://github.com/davidmallasen/LiveChess2FEN/releases) into the
"predictions" folder. You should have 5 test images and a "boards.fen" file.

4. Edit "test_lc2fen.py" and set the `ACTIVATE_*`, `MODEL_PATH_*`,
 `IMG_SIZE_*`, and `PRE_INPUT_*` constants.

   - `ACTIVATE_KERAS = True` will select Keras with tensorflow backend as the
   inference engine. The Keras engine is the slowest of the three.

   - `ACTIVATE_ONNX = True` will select ONNX Runtime as the inference engine.
   It is significantly faster than Keras but almost just as accurate. It is the
   recommended choice for any non-Jetson computer.
   
   - `ACTIVATE_TRT = True` will select TensorRT as the inference engine. It is
   the fastest of the three but only available on Jetson computers.

5. Run the "test_lc2fen.py" script.

6. You can then use LiveChess2FEN by repeating steps 6 and 7 with the
"lc2fen.py" program instead of the "test_lc2fen.py" script. Run
`python3 lc2fen.py -h` to display the help message.

## Contributing

Contributions are very welcome! Please check the 
[CONTRIBUTING](CONTRIBUTING.md) file for more information on how to
 contribute to LiveChess2FEN.

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
