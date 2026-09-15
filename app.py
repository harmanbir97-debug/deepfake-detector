import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load trained model
model = tf.keras.models.load_model(
    "model/deepfake_detector_model.keras"
)


# Page settings
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)


st.title("🔍 Deepfake Detector")
st.write("Upload an image to check whether it is REAL or FAKE.")


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prepare image
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "FAKE"
        confidence = prediction * 100
    else:
        label = "REAL"
        confidence = (1 - prediction) * 100

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")