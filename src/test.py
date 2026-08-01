
import torch 
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)  

from src.config import DEVICE, MODEL_PATH 
from src.dataset import test_loader 
from src.model import model  

model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE)) 
model.to(DEVICE) 
model.eval()  

y_true = [] 
y_pred = [] 

with torch.no_grad(): 

    for images, labels in test_loader: 

        images = images.to(DEVICE) 

        outputs = model(images) 

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())

        y_pred.extend(predicted.cpu().numpy())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"]
)) 