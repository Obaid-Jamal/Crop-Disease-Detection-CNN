import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="AI-based Crop Disease Detection", layout="centered")

# Load model
model = load_model("crop_model.h5")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>🌿 AI-based Crop Disease Detection using Leaf Images</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("<p style='text-align: center;'>Upload a leaf image to check whether it is healthy or diseased</p>", unsafe_allow_html=True)

st.write("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image (FIXED warning here)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, 128, 128, 3)

    # Prediction
    prediction = model.predict(img_resized)

    st.write("---")

    if prediction > 0.5:
        st.error("🚨 Diseased Leaf Detected")
        st.info("⚠️ The plant may be affected by disease. Please take necessary action.")
    else:
        st.success("✅ Healthy Leaf")
        st.info("🌱 The plant appears to be healthy.")

# Footer
st.write("---")