import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Path to your TFLite model
MODEL_PATH = "emtech2_project/COVID19_Xray_Detection.tflite"
IMG_SIZE = (224, 224)

# Class labels for prediction
class_labels = [
    "NORMAL",
    "PNEUMONIA"
]

# Load the TFLite model only once using Streamlit cache
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Retrieve input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image to match model input format
def preprocess_image(image_bytes):
    img = load_img(BytesIO(image_bytes), color_mode="grayscale", target_size=IMG_SIZE)
    img_array = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 1)

# Perform prediction using the TFLite interpreter
def predict(image_bytes):
    input_data = preprocess_image(image_bytes)
    input_data = input_data.astype(input_details[0]["dtype"])
    
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # Since it's binary sigmoid (1 neuron output), apply threshold
    confidence = float(output_data)
    index = int(confidence > 0.5)
    
    return index, confidence * 100

# Streamlit user interface
st.set_page_config(page_title="🩻 COVID-19 X-ray Classifier", layout="centered")
st.title("🩻 COVID-19 Chest X-ray Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(BytesIO(image_bytes), caption="Uploaded Image", use_container_width=True)

    index, confidence = predict(image_bytes)

    st.markdown(f"### 🧠 Prediction: **{class_labels[index]}**")
    st.markdown(f"### 🔬 Confidence: **{confidence:.2f}%**")
