import os
import cv2
import numpy as np
import altair as alt
from PIL import Image
import streamlit as st
from datetime import datetime
from ultralytics import YOLO

# Set a custom theme
st.set_page_config(
    page_title="Vehicle Type Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f9;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        color: #777777;
        margin-top: 30px;
        font-size: 0.9rem;
    }
    .footer a {
        color: #1f4e79;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Section
st.markdown(
    """
    <div class="title">Vehicle Type Detection Model</div>
    <div class="subtitle">Identify vehicles quickly and accurately with AI</div>
    """,
    unsafe_allow_html=True,
)

# Description Section
st.markdown(
    """
    - **Supported Vehicle Classes**: Bus, Car, Ambulance, Truck, Motorcycle
    - **Technology**: Powered by advanced artificial intelligence (YOLOv8)
    - **Developer**: Sahil S Khan
    """
)

# YOLO Model Initialization
model = YOLO("runs/best.pt")

# File Uploader Section
st.markdown("### Upload an Image to Get Started")
uploaded_file = st.file_uploader(
    "Choose an image file (png, jpg, jpeg):", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True, channels="RGB")

# Detection Button
if st.button("🚀 Detect Vehicle"):
    if uploaded_file is None:
        st.warning("Please upload an image before running detection.")
    else:
        st.markdown("### Detection Results")
        st.info("Detecting vehicles... Please wait.")

        # Run YOLO model prediction
        with st.spinner("Processing Image..."):
            result_img = model.predict(img, imgsz=640, conf=0.25)
            arry_img_result = result_img[0].plot()

            # Convert to RGB for display
            image_rgb = cv2.cvtColor(arry_img_result, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Model Prediction(s)", use_column_width=True)

# Footer Section
st.markdown(
    """
    <hr style="border-top: 1px solid #bbb;">
    <div class="footer">
        ✨ Developed by <a href="#">Sahil S Khan</a> | Powered by <strong>YOLOv8</strong> 🚀
    </div>
    """,
    unsafe_allow_html=True,
)