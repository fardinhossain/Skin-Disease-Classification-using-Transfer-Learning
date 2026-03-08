# 🩺 Skin Disease Classification Using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.72%25-brightgreen?style=for-the-badge)](/)

A deep learning-based skin disease classification system using transfer learning with **GoogLeNet** and **MobileNetV2** architectures. This project achieves **98.72% accuracy** on classifying 8 types of infectious skin diseases from clinical images.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Supported Disease Classes](#supported-disease-classes)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Citation](#citation)

---

## 🔍 Overview

This project implements a CNN-based skin disease classification system leveraging transfer learning to accurately identify 8 types of infectious skin conditions. By fine-tuning pre-trained models on ImageNet, the system achieves high accuracy even with a limited dataset of ~1,000 training images.

The project provides:
- Two pre-trained model options (GoogLeNet for accuracy, MobileNetV2 for efficiency)
- Ensemble prediction combining both models
- Comprehensive training pipeline with data augmentation
- Visualization tools for predictions and model evaluation

---

## 🎯 Problem Statement

Many people lack easy access to dermatologists for skin condition diagnosis. This project aims to provide an automated screening tool that can:

1. Classify skin diseases accurately from clinical photographs
2. Work with limited training data using transfer learning
3. Offer a lightweight model suitable for mobile/edge deployment

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Transfer Learning** | Fine-tuned GoogLeNet and MobileNetV2 pre-trained on ImageNet |
| **Multi-Class Classification** | Supports 8 infectious skin disease categories |
| **Ensemble Prediction** | Combines both models for improved accuracy |
| **Data Augmentation** | Random flips, rotations, color jitter, and affine transforms |
| **Confidence Scores** | Probability distribution across all classes |
| **Disease Information** | Provides type and description for each prediction |
| **Visualization** | Bar charts showing class probabilities |
| **GPU Acceleration** | CUDA support for faster training and inference |

---

## 🦠 Supported Disease Classes

The model classifies the following 8 skin conditions:

| Category | Disease | Code |
|----------|---------|------|
| **Bacterial** | Cellulitis | BA-cellulitis |
| **Bacterial** | Impetigo | BA-impetigo |
| **Fungal** | Athlete's Foot | FU-athlete-foot |
| **Fungal** | Nail Fungus | FU-nail-fungus |
| **Fungal** | Ringworm | FU-ringworm |
| **Parasitic** | Cutaneous Larva Migrans | PA-cutaneous-larva-migrans |
| **Viral** | Chickenpox | VI-chickenpox |
| **Viral** | Shingles | VI-shingles |

---

## 📊 Model Performance

| Model | Test Accuracy | Parameters | Best For |
|-------|---------------|------------|----------|
| **GoogLeNet** | 98.72% | ~6.6M | Maximum accuracy |
| **MobileNetV2** | 97.86% | ~3.4M | Mobile/edge deployment |
| **Ensemble** | ~99%+ | Combined | Production systems |

Both models were trained with identical hyperparameters:
- **Epochs:** 25
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, Pillow
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn
- **Progress Tracking:** tqdm

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skin-disease-classification.git
   cd skin-disease-classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## 🚀 Usage

### Training Models

Train both GoogLeNet and MobileNetV2 models:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Apply data augmentation
- Train both models for 25 epochs
- Save the best checkpoints as `GoogLeNet_best.pth` and `MobileNetV2_best.pth`
- Generate training curves and confusion matrices

### Making Predictions

**Using Ensemble (Recommended):**
```bash
python predict.py --image path/to/skin_image.jpg --model ensemble
```

**Using GoogLeNet Only:**
```bash
python predict.py --image path/to/skin_image.jpg --model googlenet
```

**Using MobileNetV2 Only:**
```bash
python predict.py --image path/to/skin_image.jpg --model mobilenet
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | Required | Path to input image |
| `--model` | `ensemble` | Model choice: `googlenet`, `mobilenet`, or `ensemble` |
| `--googlenet_path` | `GoogLeNet_best.pth` | Path to GoogLeNet checkpoint |
| `--mobilenet_path` | `MobileNetV2_best.pth` | Path to MobileNetV2 checkpoint |

### Example Output

```
Using device: cuda
Processing image: test_image.jpg

Loading models for ensemble prediction...
Loaded googlenet model with best accuracy: 98.72%
Loaded mobilenet model with best accuracy: 97.86%

============================================================
PREDICTION RESULT
============================================================
Disease: FU-ringworm
Type: Fungal
Confidence: 96.45%
Description: A fungal infection causing circular, ring-shaped rashes on the skin.
============================================================
```

---

## 📁 Project Structure

```
skin_disease_project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train.py                     # Training script
├── predict.py                   # Inference script
├── GoogLeNet_best.pth          # Trained GoogLeNet model
├── MobileNetV2_best.pth        # Trained MobileNetV2 model
├── dataset/
│   ├── train_set/              # Training images
│   │   ├── BA-cellulitis/
│   │   ├── BA-impetigo/
│   │   ├── FU-athlete-foot/
│   │   ├── FU-nail-fungus/
│   │   ├── FU-ringworm/
│   │   ├── PA-cutaneous-larva-migrans/
│   │   ├── VI-chickenpox/
│   │   └── VI-shingles/
│   ├── test_set/               # Test images
│   │   └── [same structure]
│   └── new_random_test/        # Additional test data
│       └── labels.csv
└── paper/
    ├── main.tex                # Research paper (LaTeX)
    ├── refs.bib                # Bibliography
    ├── figures/                # Paper figures
    └── README.txt              # Paper build instructions
```

---

## 📊 Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/) and contains clinical images of 8 infectious skin diseases.

| Split | Images | Classes |
|-------|--------|---------|
| Training | 925 | 8 |
| Test | 236 | 8 |

**Class Distribution:**

| Class | Training | Test |
|-------|----------|------|
| BA-cellulitis | 136 | 35 |
| BA-impetigo | 80 | 20 |
| FU-athlete-foot | 124 | 32 |
| FU-nail-fungus | 129 | 33 |
| FU-ringworm | 90 | 23 |
| PA-cutaneous-larva-migrans | 100 | 25 |
| VI-chickenpox | 133 | 34 |
| VI-shingles | 133 | 34 |

---

## 🏗️ Architecture

### Model Pipeline

```
Input Image (224x224x3)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  Resize, Normalize (ImageNet stats)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   CNN Backbone    │  GoogLeNet or MobileNetV2 (frozen early layers)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Custom Classifier │  Dropout → FC(512) → ReLU → Dropout → FC(8)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Softmax       │  8-class probability distribution
└───────────────────┘
        │
        ▼
    Prediction + Confidence Score
```

### Transfer Learning Strategy

1. **Backbone:** Pre-trained on ImageNet (1000 classes)
2. **Frozen Layers:** Early convolutional layers retain general features
3. **Fine-tuned Layers:** Last few blocks + new classifier head
4. **New Classifier:** Custom fully-connected layers for 8-class output

---

## 🔮 Future Improvements

- [ ] **Web Application:** Build a Flask/FastAPI web interface
- [ ] **Mobile App:** Deploy MobileNetV2 on Android/iOS using ONNX/TensorFlow Lite
- [ ] **Additional Classes:** Expand to more skin conditions
- [ ] **Attention Mechanisms:** Implement Grad-CAM for interpretability
- [ ] **Data Expansion:** Collect more diverse training images
- [ ] **Cross-validation:** Implement k-fold cross-validation
- [ ] **API Deployment:** RESTful API for clinical integration
- [ ] **Real-time Detection:** Webcam/camera-based live prediction

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where applicable
- Write unit tests for new features

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## ‍💻 Author

**Md. Fardin Hossain**

- **Email:** fardin1427fm@gmail.com
- **Institution:** Department of CSE, Southeast University, Dhaka, Bangladesh
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{hossain2026skin,
  title={Skin Diseases Classification Using Transfer Learning Based Convolutional Neural Networks},
  author={Hossain, Md. Fardin},
  journal={[Conference/Journal Name]},
  year={2026},
  institution={Southeast University, Bangladesh}
}
```

---

## 🙏 Acknowledgments

- Dataset provided by [Subir Biswas on Kaggle](https://www.kaggle.com/)
- Pre-trained models from [PyTorch/torchvision](https://pytorch.org/vision/)
- Inspired by research in medical image analysis

---

<p align="center">
  <b>⭐ If this project helped you, please consider giving it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for accessible healthcare
</p>
