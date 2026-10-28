import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # Загружаем предобученную VGG19
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Заменяем последний слой на 10 классов
        vgg19.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        
        # Сохраняем модель как атрибут класса
        self.vgg = vgg19

    def forward(self, x):
        return self.vgg(x)

if __name__ == '__main__':
    model = VGG()