
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Paths
MODEL_PATH = Path("best_resnet18_cat_dog.pth")
IMAGE_DIR = Path("test_images")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing: must match training preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ResNet-18 architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
classes = checkpoint["class_names"]

print("Validation accuracy:", checkpoint["validation_accuracy"])
print("Classes:", classes)
model.to(device)
model.eval()

# Class labels: confirm these match your training code
classes = ["cat", "dog"]

# Predict images
for image_path in IMAGE_DIR.iterdir():
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    print(f"Image: {image_path.name}")
    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    print("-" * 30)
