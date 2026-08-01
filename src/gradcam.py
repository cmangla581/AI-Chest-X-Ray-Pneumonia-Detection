
'''
The moodel can predict the presence of pneumonia and also the confidence of it percentage 

GRAD - CAM creates a heatmap that highlights the regions of the X ray model focused on. 
''' 

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import DEVICE, MODEL_PATH, IMAGE_SIZE
from src.model import model

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

target_layers = [model.layer4[-1]]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def generate_gradcam(image_path, output_path="gradcam_result.jpg"):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    rgb_image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    print(f"Grad-CAM image saved to: {output_path}") 
    return output_path 


if __name__ == "__main__":

    image_path = "data/test/PNEUMONIA/person1_virus_6.jpeg"

    generate_gradcam(image_path)