#!/bin/bash

conda create -y -n npbg python=3.6
conda activate npbg

pip install \
    numpy \
    pyyaml \
    torch==1.3 \
    torchvision==0.4.1 \
    tensorboardX \
    munch \
    scipy \
    matplotlib \
    Cython \
    PyOpenGL \
    PyOpenGL_accelerate \
    trimesh \
    huepy \
    "pillow<7" \
    tqdm \
    scikit-learn

conda install opencv

# need to install separately
pip install \
    git+https://github.com/DmitryUlyanov/glumpy \
    numpy-quaternion

# pycuda
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin
./configure.py --cuda-enable-gl
python setup.py install
cd ..
