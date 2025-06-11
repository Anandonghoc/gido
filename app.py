import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# Link Google Drive
file_id = "1ELXWxH1IJ525FM1F4niKr6R1x0pR2pAn"
output_model = "fer_emotion_model.h5"

# Tải model nếu chưa có
if not os.path.exists(output_model):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_model, quiet=False)

# Load model
model = tf.keras.models.load_model(output_model)

# Mapping label
class_names = ['negative', 'neutral', 'positive']

# Giao diện
st.title("Emotion Detection App")
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=200)

        img_resized = img.resize((48, 48))
        img_array = np.array(img_resized)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = gray / 255.0
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        prediction = model.predict(gray)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Predicted Emotion: **{predicted_class}**")
