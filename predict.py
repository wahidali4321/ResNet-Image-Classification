
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model checkpoint
MODEL_PATH = Path("best_resnet18_cat_dog.pth")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# Recreate the same model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

class_names = checkpoint["class_names"]

# Use the same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    print("Image:", image_path.name)
    print("Prediction:", predicted_class.capitalize())
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    image_path = input(
        "Enter the full path to a cat or dog image: "
    ).strip().strip('"')

    predict_image(image_path)
