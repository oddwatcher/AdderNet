# models/mnist_addernet_train.py
import torch.nn as nn
from models.adder_trainable import Adder2D



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.adder = Adder2D(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.adder(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MNISTAdderNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = BasicBlock(1, 16, 3, 1, 1)
        self.layer2 = BasicBlock(16, 32, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
