
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from src.config import DEVICE, MODEL_PATH
from src.dataset import test_loader
from src.model import model

# ----------------------------------------
# Output Folder
# ----------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Load Trained Model
# ----------------------------------------

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully!")

# ----------------------------------------
# Evaluation
# ----------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

# ----------------------------------------
# Classification Report
# ----------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=["NORMAL", "PNEUMONIA"]
)

print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NORMAL","PNEUMONIA"],
    yticklabels=["NORMAL","PNEUMONIA"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

print("Confusion Matrix Saved!")

# ----------------------------------------
# ROC Curve
# ----------------------------------------

fpr, tpr, _ = roc_curve(
    all_labels,
    all_probabilities
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )
)

plt.close()

print("ROC Curve Saved!")

print("\nEvaluation Completed Successfully!")