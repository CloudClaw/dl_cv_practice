import torchvision
import torch.nn as nn

from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    model = ResNet()