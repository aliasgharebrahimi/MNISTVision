# MNISTVision
> A handwritten digit recognition model featuring robust optimization, very high speed and accuracy, and an architecture of my own design.

## Overview

Recognizing handwritten digits accurately and efficiently requires a model that can achieve strong classification performance while maintaining low inference latency and computational cost.

This project focuses on developing an optimized deep learning model for handwritten digit recognition, with an emphasis on high accuracy, fast inference, and efficient computation.

The model is designed to provide reliable predictions while maintaining high execution speed, making it suitable for real-time and resource-efficient applications.

## Key Features

- Maximum execution speed
- Highly robust optimization
- Very high precision
- Advanced hyperparameter tuning with the experimental tracking method with W&B
- The modular nature of the project
- Highly professional and clean commits and branches
- Custom neural network architecture
- Includes research and scientific rationale for the entire project
- Professional README.md

## Results

MNISTVision was evaluated on the MNIST test set to measure its
classification performance, inference efficiency, and computational cost.

### Performance

| Metric        |  Score |
|---------------|-------:|
| Test Accuracy |    98% |
| Test Loss     | 0.056807 |

### Efficiency

MNISTVision is designed to provide fast inference while maintaining
a lightweight computational footprint.

| Metric | Value |
|---|------:|
| Training + Testing Time |   52s |
| Parameters | 3,426 |
| Model Size |  8 MB |

### Training Curves

The following curves illustrate the model's learning progress throughout
the training process, including changes in loss and accuracy across epochs.

![Training Curves](docs/images/training_curves.png)

### Confusion Matrix

The confusion matrix provides a detailed view of the model's
classification performance across all ten digit classes.

![Confusion Matrix](docs/images/confusion_matrix.png)

### Sample Predictions

The following examples demonstrate the model's predictions on unseen
handwritten digit images.

![Sample Predictions](docs/images/sample_predictions.png)

## Project Structure

```text
MNISTVision/
├── data/
│
├── research/
│   └── hyperparameter/
│       ├── batch_size.md
│       └── lr.md
│
├── wandb/
│
├── config.py
├── dataloader.py
├── dataset.py
├── eval.py
├── LICENSE
├── loss.py
├── neural_network.py
├── optimizer.py
├── README.md
├── requirements.txt
├── train.py
├── train_pipeline.py
├── transforms.py
└── webcam_inference.py
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MNISTVision
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify the Installation

```bash
python --version
pip --version
```

### 7. Run the Project

```bash
python train_pipeline.py
```

## Dataset

MNISTVision uses the **MNIST dataset** for handwritten digit classification.

The dataset consists of grayscale images of handwritten digits from **0 to 9**.

| Property | Value |
|---|---:|
| Training Samples | 60,000 |
| Evaluation Samples | 10,000 |
| Number of Classes | 10 |
| Image Size | 28 × 28 |
| Channels | 1 (Grayscale) |

### Preprocessing

Both training and evaluation samples are converted to PyTorch tensors using `ToTensor()`.

- Training transform: `ToTensor()`
- Evaluation transform: `ToTensor()`

The dataset is automatically downloaded to the `./data` directory when it is not already available locally.