# Imports
from torchvision.datasets import MNIST
from transforms import train_transforms, eval_transforms

# Load the training dataset and apply training transforms
train_data = MNIST(
    root="./data",
    train=True,
    transform=train_transforms,
    target_transform=None,
    download=True
)

# Load the test dataset and apply evaluation transforms
eval_data = MNIST(
    root="./data",
    train=False,
    transform=eval_transforms,
    target_transform=None,
    download=True
)