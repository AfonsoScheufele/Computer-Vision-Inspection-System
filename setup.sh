#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/train/normal data/train/defect
mkdir -p data/val/normal data/val/defect
mkdir -p data/test

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your images to data/train/normal/ and data/train/defect/"
echo "2. Add validation images to data/val/normal/ and data/val/defect/"
echo "3. Run: python src/train.py --train_dir data/train --val_dir data/val"

