import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for label, category in enumerate(["normal", "defect"]):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                raise FileNotFoundError(f"Missing class folder: {category_path}")

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if not os.path.isfile(img_path):
                    continue

                ext = os.path.splitext(img_name)[1].lower()
                if ext not in VALID_IMAGE_EXTENSIONS:
                    continue

                self.image_paths.append(img_path)
                self.labels.append(label)

        if not self.image_paths:
            raise ValueError(f"No valid images found in: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image file: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])