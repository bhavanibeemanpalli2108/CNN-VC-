import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model
from src.data_loader import create_generators

model = load_model(
    "models/vehicle_cnn.h5"
)

_, _, test_gen = create_generators()

predictions = model.predict(
    test_gen
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_gen.classes

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(
            test_gen.class_indices.keys()
        )
    )
)