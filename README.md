🩺 AI Chest X-Ray Disease Detection using Deep Learning

📌 Project Overview

This project is an AI-powered Chest X-Ray Disease Detection System developed using PyTorch and ResNet18. The application classifies chest X-ray images into two categories:

✅ NORMAL
✅ PNEUMONIA

The project also integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of the model's predictions, making the AI system more transparent and interpretable. 

📂 Project Structure
AI-Chest-XRay-Disease-Detection/
│
├── assets/
│   ├── css/
│   └── images/
│
├── models/
│   └── best_model.pth
│
├── outputs/
│   └── Grad-CAM Images
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── evaluate.py
│   ├── test.py
│   └── utils.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore 

🧠 Deep Learning Model
Architecture
ResNet18
Transfer Learning
PyTorch Framework
Classes
NORMAL
PNEUMONIA
Explainability
Grad-CAM 

📊 Technologies Used
Python
PyTorch
Torchvision
OpenCV
NumPy
Pillow
Matplotlib
Streamlit
Grad-CAM
Git
GitHub

