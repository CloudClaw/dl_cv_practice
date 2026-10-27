import torch.nn as nn
import torchvision.models as models

def get_resnet():
    model = models.resnet18(weights=None)
    
    # Меняем последний слой под 10 классов
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model