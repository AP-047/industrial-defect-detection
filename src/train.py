import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DefectDataset, get_transforms
from model import get_model

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def train():
    transform = get_transforms()

    train_dataset = DefectDataset("data/train", transform)
    val_dataset = DefectDataset("data/val", transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = get_model().to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {total_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), "outputs/model.pth")


if __name__ == "__main__":
    train()