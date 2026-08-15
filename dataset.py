from torchvision.datasets import MNIST
from transforms import train_transforms, eval_transforms


train_data = MNIST(
    root="./data",
    train=True,
    transform=train_transforms,
    target_transform=None,
    download=True
)

eval_data = MNIST(
    root="./data",
    train=False,
    transform=eval_transforms,
    target_transform=None,
    download=True
)