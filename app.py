import cv2
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model

model = load_model(
    "models/vehicle_cnn.h5"
)

classes = [
    "bus",
    "car",
    "motorcycle",
    "truck"
]

st.set_page_config(
    page_title="Vehicle Classifier",
    layout="centered"
)

st.title(
    "🚗 Vehicle Type Classification"
)

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        1
    )

    st.image(
        image,
        channels="BGR"
    )

    img = cv2.resize(
        image,
        (128,128)
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img
    )

    idx = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    st.success(
        f"Prediction: {classes[idx]}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )