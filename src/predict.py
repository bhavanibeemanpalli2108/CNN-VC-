import cv2
import numpy as np

from tensorflow.keras.models import load_model

classes = [
    "bus",
    "car",
    "motorcycle",
    "truck"
]

model = load_model(
    "models/vehicle_cnn.h5"
)


def predict_vehicle(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(
        image,
        (128,128)
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image
    )

    class_id = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    print(
        f"Vehicle: {classes[class_id]}"
    )

    print(
        f"Confidence: {confidence*100:.2f}%"
    )


predict_vehicle(
    "sample.jpg"
)