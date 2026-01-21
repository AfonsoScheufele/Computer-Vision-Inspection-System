# Computer Vision Inspection System

Automated inspection system using computer vision and deep learning for real-time defect detection.

## 🚀 Tech Stack
- TensorFlow/Keras for deep learning models
- OpenCV for image processing
- Python for complete pipeline

## ✨ Features
- Data preprocessing and augmentation
- Defect detection/classification models
- Real-time inference pipeline
- Results visualization and metrics

## 📊 Applications
- Industrial quality control
- Automated product inspection
- Image anomaly detection

## 📁 Project Structure
```
Computer-Vision-Inspection-System/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── models/
├── notebooks/
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## 🛠️ Installation

**Quick install (recommended):**
```bash
./install_dependencies.sh
```

**Or manually:**
```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 -m pip install --user --break-system-packages packaging pyparsing python-dateutil
```

## 🎯 Usage

### Training
```bash
python3 src/train.py --model resnet50 --train_dir data/train --val_dir data/val
```

### Inference

**Single image:**
```bash
python3 src/inference.py --model models/best_model_resnet50.h5 --image path/to/image.jpg --visualize
```

**Directory of images:**
```bash
python3 src/inference.py --model models/best_model_resnet50.h5 --dir pasta_com_imagens/ --output resultados/
```

**Note:** Always use `python3` instead of `python`

## 📈 Model Performance
- Accuracy: TBD
- Precision: TBD
- Recall: TBD
- F1-Score: TBD
