#!/usr/bin/env bash

sudo apt-get install -y cmake
# higher version is needed only for onnxsim >= 0.4.13
# sudo apt install cmake>=3.22

# Dependencies for building pillow-simd
pip3 install -r requirements_modelmaker.txt

# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd
