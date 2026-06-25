"""
Run this script ONCE on your local machine (or in Google Colab) to create
the babesia_cnn_softmax_model.keras file needed by app.py.

Usage:
    pip install tensorflow numpy
    python generate_model_locally.py

This builds the SAME architecture used in the Colab notebook but with
random (untrained) weights — useful ONLY for testing that app.py loads
and runs correctly before you have a real trained model.

For the REAL model: run Babesia_CNN_SoftMax_Colab.ipynb in Google Colab
(Runtime → T4 GPU), then download babesia_cnn_softmax_model.keras from
Section 7, and place it next to app.py.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 128
NUM_CLASSES = 2

def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    return models.Model(inputs, outputs, name="Babesia_CNN_SoftMax")


model = build_custom_cnn()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

output_path = "babesia_cnn_softmax_model.keras"
model.save(output_path)
print(f"✅ Model saved to: {output_path}")
print(f"   (This is an UNTRAINED model — predictions will be random.)")
print(f"   For real predictions, run the Colab notebook and use that .keras file.")
