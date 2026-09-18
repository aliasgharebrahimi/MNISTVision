import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1, kaiming_normal = False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        if kaiming_normal:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu", a=0.01)

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = self.pool(x)

        return x