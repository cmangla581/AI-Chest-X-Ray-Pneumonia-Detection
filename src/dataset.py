
# The dataset py code can be presented as: 

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if __name__ == "__main__":
    print("Training Images :", len(train_dataset))
    print("Validation Images:", len(val_dataset))
    print("Testing Images :", len(test_dataset))

    print("\nClasses:", train_dataset.classes) 

#  This last statement id for verifying 