import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# ✅ Image input settings
IMG_SIZE = (224, 224)

# ✅ Label classes
class_labels = [
    "COVID-19", 
    "Lung Opacity", 
    "Normal", 
    "Viral Pneumonia"
]

# ✅ Load model (from local .h5 file in same folder or subfolder)
@st.cache_resource
def load_trained_model():
    return load_model("COVID19_Xray_Detection.h5", compile=False)

model = load_trained_model()

# ✅ Image pre-processing
def preprocess_image(image_bytes):
    img = load_img(BytesIO(image_bytes), target_size=IMG_SIZE)
    img_array = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Inference
def predict(image_bytes):
    processed = preprocess_image(image_bytes)
    preds = model.predict(processed)[0]
    index = np.argmax(preds)
    confidence = preds[index] * 100
    return index, confidence

# ✅ Streamlit page setup
st.set_page_config(page_title="🩻 COVID-19 X-ray Classifier", layout="centered")
st.title("🩻 COVID-19 Chest X-ray Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(BytesIO(image_bytes), caption="Uploaded Image", use_container_width=True)

    index, confidence = predict(image_bytes)

    st.markdown(f"### 🧠 Prediction: **{class_labels[index]}**")
    st.markdown(f"### 🔬 Confidence: **{confidence:.2f}%**")
