
'''
train.py : learns from thousands of images 
test.py : evaluates the model on test dataset 

predict.py : predicts the class of one new X ray image 

''' 

# imprting the libraries  
import torch 
from torchvision import transforms 
from PIL import Image 

from src.config import DEVICE, MODEL_PATH, IMAGE_SIZE  
from src.model import model  

model.load_state_dict(
    torch.load(MODEL_PATH, map_location = DEVICE)
) 

model.to(DEVICE) 
model.eval() 

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor()
]) 

classes = [
    "NORMAL", 
    "PNEUMONIA"
] 

def predict_image(image_path): 

    image = Image.open(image_path).convert("RGB")  

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return (
        classes[predicted.item()],
        confidence.item() * 100
    )


if __name__ == "__main__":

    image_path = "data/test/NORMAL/IM-0001-0001.jpeg"

    prediction, confidence = predict_image(image_path)

    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")