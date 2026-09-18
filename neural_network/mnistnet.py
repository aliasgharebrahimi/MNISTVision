import torch.nn as nn
from torchinfo import summary
from blocks import ConvBlock

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            kaiming_normal=True
        )

        self.block2 = ConvBlock(
            in_channels=6,
            out_channels=12,
            kernel_size=3
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            in_features=300,
            out_features=10
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x

model = MNISTNet()

summary(model, input_size=(1, 1, 28, 28))