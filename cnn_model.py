import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convs (each BN+ReLU) followed by a 2x2 max pool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.stage2 = ConvBlock(32, 64)
        self.stage3 = ConvBlock(64, 128)
        self.stage4 = ConvBlock(128, 256)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    model = CNN(num_classes=10)
    dummy_input = torch.randn(8, 3, 32, 32)  # batch of 8 RGB 32x32 images
    output = model(dummy_input)
    print(output.shape)  # expected: torch.Size([8, 10])
