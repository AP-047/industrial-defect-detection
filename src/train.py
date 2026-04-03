import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import DefectDataset, get_transforms
from model import get_model

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train(train_dir, val_dir, batch_size, lr, epochs, output_model, seed):
    set_seed(seed)

    transform = get_transforms()

    train_dataset = DefectDataset(train_dir, transform)
    val_dataset = DefectDataset(val_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = get_model().to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=lr)

    for epoch in range(epochs):
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

    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), output_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train defect detection model.")
    parser.add_argument("--train-dir", default="data/train", help="Path to training data directory.")
    parser.add_argument("--val-dir", default="data/val", help="Path to validation data directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--output-model", default="outputs/model.pth", help="Path to save trained model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_model=args.output_model,
        seed=args.seed,
    )