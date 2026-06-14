from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.plot_results import (
    plot_accuracy,
    plot_loss
)


from src.model import build_model
from src.data_loader import create_generators

model = build_model()

train_gen, val_gen, test_gen = create_generators()

# callbacks = [

#     EarlyStopping(
#         monitor='val_loss',
#         patience=5,
#         restore_best_weights=True
#     ),

#     ModelCheckpoint(
#         "models/vehicle_cnn.h5",
#         save_best_only=True
#     ),

#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.1,
#         patience=3
#     )
# ]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=300,
    # callbacks=callbacks
)

model.save("models/vehicle_cnn.h5")


plot_accuracy(history)
plot_loss(history)