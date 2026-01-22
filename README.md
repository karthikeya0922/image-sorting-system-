# 🗑️ AI-Based Smart Waste Detection System Using Webcam

A simple, beginner-friendly computer vision project that uses your laptop's webcam to classify waste items in real-time. Perfect for learning AI, computer vision, and building practical ML applications!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Waste Categories](#waste-categories)
- [System Workflow](#system-workflow)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project demonstrates how to build an AI-powered waste classification system using:
- **Your laptop's webcam** for real-time detection
- **Transfer learning** with MobileNetV2 for efficient training
- **PyTorch** for deep learning
- **OpenCV** for webcam capture and display

**No hardware required** - just your laptop and webcam!

---

## ✨ Features

✅ **Real-time Detection**: Classify waste items instantly using your webcam  
✅ **6 Waste Categories**: Plastic, Paper, Metal, Glass, Organic, Non-recyclable  
✅ **Beginner-Friendly**: Clean, commented code with step-by-step guide  
✅ **CPU Compatible**: Runs on any laptop without GPU  
✅ **Lightweight**: ~15MB model, trains in 15-30 minutes  
✅ **Visual Feedback**: Display predictions with confidence scores  
✅ **Pre-trained Model**: Option to use provided weights or train your own  

---

## 🗂️ Waste Categories

The system classifies waste into **6 categories**:

| Category | Examples | Color Code |
|----------|----------|------------|
| 🔵 **Plastic** | Bottles, bags, containers | Blue |
| 📄 **Paper** | Newspapers, cardboard, magazines | Brown |
| 🔘 **Metal** | Cans, foil, metal containers | Gray |
| 💚 **Glass** | Bottles, jars | Green |
| 🟢 **Organic** | Food waste, plant matter | Dark Green |
| ⚫ **Non-recyclable** | Mixed waste, contaminated items | Black |

---

## 🔄 System Workflow

```mermaid
graph LR
    A[Webcam Feed] --> B[Capture Frame]
    B --> C[Preprocess Image]
    C --> D[MobileNetV2 Model]
    D --> E[Class Prediction]
    E --> F[Display Result]
    F --> A
    
    style A fill:#e1f5ff
    style D fill:#ffe1e1
    style F fill:#e1ffe1
```

**Step-by-step process**:
1. **Capture**: Webcam captures live video frames
2. **Preprocess**: Resize to 224×224, normalize pixel values
3. **Inference**: MobileNetV2 predicts waste category
4. **Display**: Show prediction + confidence on screen

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or USB)
- ~2GB free disk space (for dataset)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/smart-waste-detection.git
cd smart-waste-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies** (automatically installed):
- `torch>=1.13.0` - Deep learning framework
- `torchvision>=0.14.0` - Computer vision utilities
- `opencv-python>=4.8.0` - Webcam capture
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization
- `albumentations>=1.3.0` - Data augmentation
- `tqdm>=4.65.0` - Progress bars
- `scikit-learn>=1.3.0` - Metrics

---

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Fast)

If you have the pre-trained model weights:

```bash
# Run webcam detection immediately
python app/webcam_detect.py --model model/checkpoints/best_model.pth
```

### Option 2: Train Your Own Model (Full Experience)

```bash
# Step 1: Download dataset (~2GB, takes 5-10 minutes)
python dataset/download_data.py

# Step 2: Organize and split data
python dataset/organize_data.py
python dataset/split_data.py

# Step 3: Train model (15-30 minutes on CPU)
python model/train.py

# Step 4: Evaluate model
python model/evaluate.py

# Step 5: Run webcam detection
python app/webcam_detect.py
```

---

## 📖 Detailed Usage

### 1. Dataset Preparation

#### Download TrashNet Dataset
```bash
python dataset/download_data.py
```
- Downloads ~2500 images from TrashNet dataset
- Saves to `dataset/raw/`
- Shows progress bar

#### Organize Dataset
```bash
python dataset/organize_data.py
```
- Organizes images into category folders
- Creates `dataset/processed/` structure

#### Create Train/Val/Test Split
```bash
python dataset/split_data.py
```
- Splits: 70% train, 15% validation, 15% test
- Reproducible with fixed seed (42)
- Applies data augmentation to training set

**Data Augmentation** (automatic):
- Horizontal flip (50% chance)
- Rotation ±15° (50% chance)
- Brightness/contrast adjustment (30% chance)
- Random crop and resize

---

### 2. Model Training

#### Basic Training
```bash
python model/train.py
```

**Default configuration**:
- Model: MobileNetV2 (pretrained on ImageNet)
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001
- Optimizer: Adam

#### Custom Training Parameters
```bash
python model/train.py --epochs 30 --batch-size 64 --lr 0.0005
```

**Training output**:
```
Epoch 1/20:
Train Loss: 1.234 | Train Acc: 65.2%
Val Loss: 0.987 | Val Acc: 72.5%

Epoch 20/20:
Train Loss: 0.234 | Train Acc: 92.3%
Val Loss: 0.456 | Val Acc: 85.7%

✅ Training complete! Best model saved to: model/checkpoints/best_model.pth
```

**Training time**:
- CPU: 15-30 minutes
- GPU: 3-5 minutes

---

### 3. Model Evaluation

```bash
python model/evaluate.py
```

**Output files** (saved to `model/results/`):
- `confusion_matrix.png` - Visual representation of predictions
- `classification_report.txt` - Precision, recall, F1 per class
- `training_history.json` - Loss and accuracy curves

**Example metrics**:
```
Overall Accuracy: 85.7%

Per-class Performance:
              precision  recall  f1-score  support
    Plastic      0.89     0.92      0.90       85
      Paper      0.85     0.81      0.83       72
      Metal      0.91     0.88      0.89       68
      Glass      0.87     0.84      0.85       75
    Organic      0.79     0.83      0.81       69
Non-recyclable   0.83     0.86      0.84       71
```

---

### 4. Webcam Detection App

#### Run Detection
```bash
python app/webcam_detect.py
```

**Keyboard controls**:
- `q` - Quit application
- `s` - Save screenshot
- `SPACE` - Pause/Resume
- `c` - Toggle confidence display

**Display information**:
- **Top-left**: Predicted class (e.g., "PLASTIC")
- **Top-right**: Confidence score (e.g., "94.5%")
- **Bottom**: FPS counter
- **Color-coded border**: Changes based on predicted class

#### Advanced Options
```bash
# Use custom model
python app/webcam_detect.py --model path/to/model.pth

# Adjust confidence threshold
python app/webcam_detect.py --threshold 0.8

# Use specific webcam (if multiple cameras)
python app/webcam_detect.py --camera 1

# Save predictions to log
python app/webcam_detect.py --save-log predictions.csv
```

---

## 📁 Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed file organization.

**Quick overview**:
```
smart-waste-detection/
├── dataset/          # Data download and processing
├── model/            # Training and evaluation
├── app/              # Webcam detection application
├── docs/             # Documentation and deliverables
├── examples/         # Test images
└── README.md         # This file
```

---

## 📊 Results

### Achieved Performance

With the provided dataset and default settings:
- **Training Accuracy**: 98.73%
- **Validation Accuracy**: 98.96%
- **Test Accuracy**: 98.93%
- **Inference Speed**: 15-30 FPS (CPU), 60+ FPS (GPU)

### Confusion Matrix Example

![Confusion Matrix](docs/diagrams/confusion_matrix_example.png)

### Sample Predictions

| Image | Prediction | Confidence |
|-------|------------|------------|
| ![Plastic](examples/plastic_bottle.jpg) | Plastic | 96.2% |
| ![Paper](examples/paper_bag.jpg) | Paper | 89.5% |
| ![Metal](examples/metal_can.jpg) | Metal | 93.7% |

---

## 🐛 Troubleshooting

### Webcam Issues

**Problem**: Webcam not detected
```bash
# Test webcam
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```
**Solution**: Try different camera index (0, 1, 2...)

**Problem**: Low FPS / Laggy
**Solutions**:
- Reduce inference frequency in `app/webcam_detect.py` (line 45)
- Lower webcam resolution
- Close other applications

### Training Issues

**Problem**: Out of memory during training
**Solutions**:
- Reduce batch size: `--batch-size 16`
- Train on smaller dataset
- Close other applications

**Problem**: Poor accuracy (<70%)
**Solutions**:
- Train for more epochs: `--epochs 30`
- Adjust learning rate: `--lr 0.0005`
- Check dataset quality

### Installation Issues

**Problem**: `torch` installation fails
**Solution**: Use CPU-only version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: `opencv-python` import error
**Solution**:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

---

## 🎓 Educational Use

This project is perfect for:
- 📚 **Students** learning AI/ML concepts
- 👨‍🏫 **Teachers** demonstrating computer vision
- 🔬 **Researchers** prototyping waste classification
- 🌍 **Environmental projects** promoting recycling

### Learning Outcomes
- Transfer learning with pretrained models
- Data preprocessing and augmentation
- Real-time computer vision applications
- Model evaluation and metrics
- Python best practices

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Open a Pull Request

**Areas for improvement**:
- Add more waste categories
- Improve model accuracy
- Add mobile app version
- Create web interface
- Multi-language support

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TrashNet Dataset**: Gary Thung & Mindy Yang
- **MobileNetV2**: Google Research
- **PyTorch Team**: For excellent documentation
- **OpenCV Community**: For computer vision tools

---

## 📞 Contact

**Project Maintainer**: K Karthikeya Gupta


---

## 🌟 Star This Project!

If you found this helpful, please ⭐ star this repository to help others discover it!

---

**Last Updated**: 2026-01-08  
**Version**: 1.1.0  
**Status**: ✅ Production Ready

