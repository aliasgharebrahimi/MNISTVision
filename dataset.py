import torch
import torchvision.datasets.MNIST as MNIST

train_data = MNIST(
    root="./data",
    train=True,
    transforms=train_transforms,
    download=True
)

eval_data = MNIST(
    root="./data",
    train=False,
    transforms=eval_transforms,
    download=True
)