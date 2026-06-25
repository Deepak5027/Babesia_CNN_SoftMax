"""
Streamlit web app for the Babesia (CNN + SoftMax) blood-smear detector.

Run locally:
    streamlit run app.py

Deploy for free:
    1. Push this file + requirements.txt + babesia_cnn_softmax_model.keras to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> select the repo -> Deploy.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "babesia_cnn_softmax_model.keras"
IMG_SIZE = 128
CLASS_NAMES = ["Normal", "Infected"]

st.set_page_config(page_title="Babesia Infection Detector", page_icon="🩸", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🩸 Automated Babesia Infection Detection")
    st.caption("CNN + SoftMax Classifier — Blood Smear Image Analysis")

    st.warning(
        "This model was trained on the public NIH/Kaggle malaria blood-smear dataset, used as a "
        "research proxy for Babesia due to the lack of public Babesia image data and the documented "
        "visual similarity between the two intra-erythrocytic parasites. It is a proof-of-concept "
        "academic tool, **not a validated clinical diagnostic device** — do not use it to make real "
        "medical decisions."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a blood smear cell image (.png / .jpg / .jpeg)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            batch = preprocess_image(image)
            probs = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx]) * 100

        st.subheader("Result")
        if pred_label == "Infected":
            st.error(f"🔴 Prediction: **{pred_label}**  (confidence: {confidence:.2f}%)")
        else:
            st.success(f"🟢 Prediction: **{pred_label}**  (confidence: {confidence:.2f}%)")

        col1, col2 = st.columns(2)
        col1.metric("P(Normal)", f"{probs[0] * 100:.2f}%")
        col2.metric("P(Infected)", f"{probs[1] * 100:.2f}%")

        st.progress(float(probs[1]))
    else:
        st.info("Upload an image above to get a prediction.")


if __name__ == "__main__":
    main()
