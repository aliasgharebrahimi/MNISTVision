# MNISTVision

> A handwritten digit recognition model featuring robust optimization, very high speed and accuracy, and an architecture of my own design.

---

## Overview

Recognizing handwritten digits accurately and efficiently requires a model that can achieve strong classification performance while maintaining low inference latency and computational cost.

This project focuses on developing an optimized deep learning model for handwritten digit recognition, with an emphasis on high accuracy, fast inference, and efficient computation.

The model is designed to provide reliable predictions while maintaining high execution speed, making it suitable for real-time and resource-efficient applications.

---

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

---

## Results

MNISTVision was evaluated on the MNIST test set to measure its classification performance, inference efficiency, and computational cost.

### Performance

| Metric | Score |
|---|---:|
| Test Accuracy | 98% |
| Test Loss | 0.056807 |

### Efficiency

MNISTVision is designed to provide fast inference while maintaining a lightweight computational footprint.

| Metric | Value |
|---|---:|
| Training + Testing Time | 52s |
| Parameters | 3,426 |
| Model Size | 8 MB |

### Training Curves

The following curves illustrate the model's learning progress throughout the training process, including changes in loss and accuracy across epochs.

![Training Curves](docs/images/training_curves.png)

### Confusion Matrix

The confusion matrix provides a detailed view of the model's classification performance across all ten digit classes.

![Confusion Matrix](docs/images/confusion_matrix.png)

### Sample Predictions

The following examples demonstrate the model's predictions on unseen handwritten digit images.

<video src="media/inderence_media/MNISTVision 2026-09-05 14-17-57.mp4" controls></video>

---

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

---

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

---

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

---

## Training

MNISTVision uses a lightweight training configuration designed for efficient handwritten digit classification.

### Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 2 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy Loss |
| Kernel Size | 3 |

### Optimization

The model is optimized using the **Adam optimizer** with a learning rate of `0.001`.

The training objective is defined using **Cross-Entropy Loss**, which is suitable for the multi-class handwritten digit classification task.

### Training Pipeline

The training process consists of iterating over the training dataset for the configured number of epochs, computing the classification loss, and updating the model parameters using Adam.

---

## Reproducibility

MNISTVision uses a fixed random seed to improve the reproducibility of training and evaluation results.

A fixed seed is configured to ensure consistent behavior across experiments involving randomized operations.

```python
SEED = 42
```

Using a fixed seed helps make experiments more consistent and allows results to be compared more reliably across different runs.

---

## Technical Details

MNISTVision uses a custom convolutional neural network designed for handwritten digit classification.

### Model Architecture

The network consists of two convolutional blocks followed by a fully connected classification layer.

| Layer | Configuration |
|---|---|
| Conv2D | 1 → 6 channels, Kernel Size = 3, Stride = 1 |
| ReLU | Activation |
| MaxPool2D | Kernel Size = 2 |
| Conv2D | 6 → 12 channels, Kernel Size = 3, Stride = 1 |
| ReLU | Activation |
| MaxPool2D | Kernel Size = 2 |
| Flatten | Feature Flattening |
| Linear | 300 → 10 |

### Classification

The final fully connected layer produces **10 output values**, corresponding to the ten MNIST digit classes (`0`–`9`).

The model uses **Cross-Entropy Loss** for multi-class classification and the **Adam optimizer** for parameter optimization.

### Parameter Count

The model contains **3,730 trainable parameters**.

---

## Experiment Tracking

Experiments and hyperparameter tuning were tracked using **Weights & Biases (W&B)**.

[View experiments on W&B](https://wandb.ai/aliasghare1-ai-emacv/MNISTVision/table?nw=nwuseraliasghare1ai)

---

## Limitations

The current implementation has several limitations:

- The model is currently evaluated only on the MNIST dataset.
- The current architecture is optimized for the relatively simple MNIST classification task and may not generalize to more complex image classification problems.

---

## Future Work

Potential improvements and extensions for MNISTVision include:

- Better and more advanced optimization
- Using other tools for experiment tracking
- Adding documents and further research

---

## Citation

If you use MNISTVision in your research or project, please cite this repository.

```bibtex
@software{mnistvision,
  title  = {MNISTVision},
  author = {Ali Asghar Ebrahimi},
  year   = {2026},
  url    = {https://github.com/aliasgharebrahimi/MNISTVision}
}
```

```bibtex
@article{mnistvision_paper,
  title   = {},
  author  = {Ali Asghar Ebrahimi},
  year    = {2026},
  journal = {},
  url     = {}
}
```

---

## License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for the full license text.

---

## Acknowledgments

This project was developed as an independent deep learning and computer vision project.

Special thanks to the open-source community and the developers of PyTorch and TorchVision for providing the tools and frameworks used in this project.

---

## Contributing

Contributions are welcome.

If you would like to contribute to MNISTVision, please:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and keep the code clean and focused.
4. Commit your changes with a clear commit message.
5. Open a Pull Request describing your changes.

Please ensure that contributions are consistent with the project's goals and maintain a clean and reproducible development workflow.