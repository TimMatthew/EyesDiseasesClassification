import torch.nn as nn
import torchvision
from torchvision.ops import Conv2dNormActivation


class MyCNN(nn.Module):

    def __init__(self, classes=4):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(16, 1e-05, 0.1, True),
            nn.SiLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(32, 1e-05, 0.1, True),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(64, 1e-05, 0.1, True),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(128, 1e-05, 0.1, True),
            nn.SiLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(256, 1e-05, 0.1, True),
            nn.SiLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.BatchNorm2d(512, 1e-05, 0.1, True),
            nn.SiLU(),
        )

        self.adaptPool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptPool(x)
        x = self.fc(x)
        return x

