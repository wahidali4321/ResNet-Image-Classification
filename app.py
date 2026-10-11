
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "best_resnet18_cat_dog.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load the trained model once
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_names"]


# Image preprocessing must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Interface
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write(
    "Upload an image and let the trained ResNet-18 model "
    "predict whether it contains a cat or a dog."
)

uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("Predict Image", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                model, class_names = load_model()

                image_tensor = transform(image).unsqueeze(0)
                image_tensor = image_tensor.to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(
                        outputs, dim=1
                    )[0]

                predicted_index = probabilities.argmax().item()
                predicted_class = class_names[predicted_index]
                confidence = (
                    probabilities[predicted_index].item() * 100
                )

            st.success(
                f"Prediction: {predicted_class.capitalize()}"
            )
            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated probability "
                "for the predicted class, not a guarantee of correctness."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
