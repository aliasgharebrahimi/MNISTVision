import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.leaky_relu(x)
            return self.pool(x)