
from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path(
    r"C:\Users\DELL\Downloads\archive\train\train"
)

BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 0.0001
DATA_FRACTION = 1.0

MODEL_PATH = Path("best_resnet18_cat_dog.pth")

random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)


# ==========================================
# 2. FIND AND SPLIT THE DATASET
# ==========================================

cat_images = list(DATA_DIR.glob("cat.*.jpg"))
dog_images = list(DATA_DIR.glob("dog.*.jpg"))

print("Cats found:", len(cat_images))
print("Dogs found:", len(dog_images))

if not cat_images or not dog_images:
    raise FileNotFoundError(
        f"No cat/dog images found in {DATA_DIR}"
    )

all_images = (
    [(path, 0) for path in cat_images]
    + [(path, 1) for path in dog_images]
)

random.shuffle(all_images)

subset_size = int(len(all_images) * DATA_FRACTION)
all_images = all_images[:subset_size]

split_index = int(0.8 * len(all_images))

train_data = all_images[:split_index]
val_data = all_images[split_index:]

print("Total selected images:", len(all_images))
print("Training images:", len(train_data))
print("Validation images:", len(val_data))


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. DATASET CLASS
# ==========================================

class CatDogDataset(Dataset):

    def __init__(self, image_data, transform=None):
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image_path, label = self.image_data[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, label


train_dataset = CatDogDataset(train_data, transform)
val_dataset = CatDogDataset(val_data, transform)


# ==========================================
# 5. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Data loaders created successfully.")


# ==========================================
# 6. PRETRAINED RESNET-18
# ==========================================

# Downloads ImageNet weights the first time.
# An internet connection is required initially.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace the original 1000-class output layer.
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Pretrained ResNet-18 initialized.")
print("Output classes: cat (0), dog (1)")


# ==========================================
# 7. TRAINING AND VALIDATION
# ==========================================

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    # ---------- Training ----------
    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_total += labels.size(0)
        train_correct += (
            predictions == labels
        ).sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Training batch "
                f"{batch_index + 1}/{len(train_loader)}"
            )

    train_loss = train_loss_total / train_total
    train_accuracy = 100 * train_correct / train_total

    # ---------- Validation ----------
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # ---------- Save best model ----------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": ["cat", "dog"],
                "validation_accuracy": val_accuracy
            },
            MODEL_PATH
        )

        print("Best model saved!")

    print("-" * 50)


# ==========================================
# 8. FINISH
# ==========================================

print("Training completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print("Best model path:", MODEL_PATH.resolve())