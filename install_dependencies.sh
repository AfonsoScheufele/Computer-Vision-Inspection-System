#!/bin/bash

echo "Installing all dependencies..."
echo "================================"

python3 -m pip install --user --break-system-packages \
    numpy \
    tensorflow \
    opencv-python \
    matplotlib \
    pillow \
    scikit-learn \
    pyyaml \
    tqdm \
    packaging \
    pyparsing \
    python-dateutil \
    contourpy \
    kiwisolver \
    fonttools \
    cycler \
    requests \
    rich

echo ""
echo "================================"
echo "Installation complete!"
echo ""
echo "Test with:"
echo "python3 -c 'import numpy, tensorflow, cv2, matplotlib; print(\"All OK\")'"

