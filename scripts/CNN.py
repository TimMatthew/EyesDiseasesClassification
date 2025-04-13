import torch.nn as nn
import torchvision
from torchvision.ops import Conv2dNormActivation, StochasticDepth


class MyCNN(nn.Module):

    def __init__(self, classes=4, baseline_maps=16):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, baseline_maps, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            nn.Conv2d(baseline_maps, baseline_maps * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps * 2, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            nn.Conv2d(baseline_maps * 2, baseline_maps * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps * 4, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            nn.Conv2d(baseline_maps * 4, baseline_maps * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps * 8, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            nn.Conv2d(baseline_maps * 8, baseline_maps * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps * 16, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            nn.Conv2d(baseline_maps * 16, baseline_maps * 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(baseline_maps * 32, 1e-05, 0.1, True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.SiLU(),

            StochasticDepth(p=0.19230769230769232, mode='row')
        )

        self.adaptPool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(baseline_maps * 32, baseline_maps * 16),
            nn.Dropout(0.3),
            nn.Linear(baseline_maps * 16, baseline_maps * 4),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(baseline_maps * 4, classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptPool(x)
        x = self.classifier(x)
        return x


