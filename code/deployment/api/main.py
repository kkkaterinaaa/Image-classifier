from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch
import io

app = FastAPI()

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

model.load_state_dict(torch.load("resnet_cifar10.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

CIFAR10_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image: Image.Image):
    image = transform(image).unsqueeze(0)
    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)

        return JSONResponse({"predicted_class": CIFAR10_CLASSES[predicted_class.item()]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
