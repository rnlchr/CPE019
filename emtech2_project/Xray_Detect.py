import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv

# Class mapping
label_to_name = {
    "0": "NORMAL",
    "1": "PNEUMONIA"
}
class_labels = list(label_to_name.keys())

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("COVID19_Xray_Detection.h5")
    return model

model = load_model()

# Streamlit UI
st.title("🩻 COVID-19 Chest X-ray Classifier")
st.markdown("Upload a chest X-ray image to detect if it shows signs of **Pneumonia** or is **Normal**.")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

# Preprocessing function
def preprocess_image(image):
    image_rgb = image.convert("RGB")
    image_np = tf.keras.utils.img_to_array(image_rgb)
    img_gray = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
    img_resized = cv.resize(img_gray, (224, 224))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, 224, 224, 1)
    return img_input

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0][0]

    # Determine result
    if prediction >= 0.5:
        label = "1"
        confidence = prediction
    else:
        label = "0"
        confidence = 1 - prediction

    predicted_name = label_to_name[label]

    # Display result
    st.markdown(f"### 🧠 Prediction: **{predicted_name}**")
    st.markdown(f"Confidence Score: **{confidence * 100:.2f}%**")
