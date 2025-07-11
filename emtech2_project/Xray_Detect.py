import os
import gdown
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Google Drive direct model download link
GDRIVE_MODEL_ID = "1kCWOATEzCVxJQhaCLz5_Em7H3ydxsvs4"  # 🔁 REPLACE THIS with your actual model ID
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/Datasets/dataset/xray_dataset_covid19/COVID19_Xray_Detection.h5"
GDRIVE_URL = f"https://drive.google.com/drive/folders/1lxVfKnJN8alequpqVWshdjKjdPRQNnx2?usp=drive_link"

# Class mapping
label_to_name = {
    0: "COVID-19",
    1: "Lung Opacity",
    2: "Normal",
    3: "Viral Pneumonia"
}

# Load model with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Downloading model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Streamlit UI
st.title("🩻 COVID-19 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify it into one of the four categories.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {label_to_name[predicted_class]}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
