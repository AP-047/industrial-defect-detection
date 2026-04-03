import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=2):
    model = model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze most layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last convolution block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model