import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)

    # Freeze backbone (faster training)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model