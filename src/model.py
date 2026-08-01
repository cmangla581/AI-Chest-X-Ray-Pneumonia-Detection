
'''
Here, instead of creating a neural network from scratch, we'll use a Transfer Learning.  

Bascally here, we use the model called ResNet - 18, that has already learned useful image features from 
millions of images and adapt it toour problem. 

''' 

import torch
import torch.nn as nn
from torchvision import models

from src.config import DEVICE

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Enable gradients for the last convolutional block
for param in model.layer4.parameters():
    param.requires_grad = True

# Enable gradients for the final classifier
for param in model.fc.parameters():
    param.requires_grad = True

# Replace the final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Move model to CPU or GPU
model = model.to(DEVICE)

if __name__ == "__main__":
    print(model)