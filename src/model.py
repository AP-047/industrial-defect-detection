import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(num_classes=2):
    # Start from ImageNet-pretrained weights to benefit from learned visual features.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze the backbone so training focuses on the new task-specific layers first.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last convolutional block to adapt high-level features to defect detection.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the original ImageNet classifier with a smaller head for binary classification.
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    # Return the adapted model ready for fine-tuning.
    return model