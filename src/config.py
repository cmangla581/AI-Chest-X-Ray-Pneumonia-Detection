
import os
import torch

# ==========================
# Project Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ==========================
# Training Parameters
# ==========================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

# ==========================
# Device Configuration
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", DEVICE)

'''
Now, the explanation of the code will be done as: 
1. TRAIN_DIR, TEST_DIR, VAL_DIR : tell the program where to find images 
2. MODEL_PATH :  where the trained model will be saved.
3. IMAGE_SIZE = 224 : matches the expected input size for pretrained models like ResNet.
4. BATCH_SIZE = 16 : a good balance for CPU training.
5. DEVICE automatically uses a GPU if available, otherwise it falls back to the CPU.

'''