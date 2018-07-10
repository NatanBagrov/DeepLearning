#!/bin/bash

pip install --user virtualenv

virtualenv --system-site-packages -p python3 tensorflow
source ~/tensorflow/bin/activate
pip3 install --user --upgrade pip
pip3 install --user --upgrade tensorflow-gpu==1.5
pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
pip3 install --user numpy scipy matplotlib keras
pip3 install pulp
pip3 install sinkhorn_knopp
pip3 install opencv-python
pip3 install sklearn

