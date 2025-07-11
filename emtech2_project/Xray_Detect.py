import os
import gdown
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ✅ Set paths and Google Drive info
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/Datasets/dataset/xray_dataset_covid19/COVID19_Xray_Detection.keras"
MODEL_DIR = os.path.dirname(MODEL_PATH)
GDRIVE_FILE_ID = "1405exPVV5Zrkq8rwKs-9h6NScRoLqWwV"
GDRIVE_URL = f"https://drive.google.com/drive/folders/1lxVfKnJN8alequpqVWshdjKjdPRQNnx2?usp=sharing"

# ✅ Class labels
label_to_name = {
    0: "COVID-19",
    1: "Lung Opacity",
    2: "Normal",
    3: "Viral Pneumonia"
}

# ✅ Load model with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ✅ Streamlit UI
st.title("🩻 COVID-19 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify it into one of the four categories.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {label_to_name[predicted_class]}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
