"""
Streamlit web app for the Babesia (CNN + SoftMax) blood-smear detector.

Run locally:
    streamlit run app.py

Deploy for free:
    1. Push this file + requirements.txt + babesia_cnn_softmax_model.keras to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app" -> select the repo -> Deploy.
"""

import os
import traceback

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
    # Diagnose the most common deployment failure before even calling load_model:
    # the model file missing, empty, or a broken Git LFS pointer instead of the real binary.
    if not os.path.exists(MODEL_PATH):
        cwd_files = os.listdir(".")
        raise FileNotFoundError(
            f"'{MODEL_PATH}' was not found in the app directory. "
            f"Files actually present here: {cwd_files}. "
            "Make sure the .keras file is committed to the repo root (same folder as app.py) "
            "and that the filename matches exactly, including case."
        )

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < 0.05:  # a real CNN .keras file will never be this small
        with open(MODEL_PATH, "rb") as f:
            head = f.read(200)
        raise ValueError(
            f"'{MODEL_PATH}' exists but is only {size_mb*1024:.1f} KB — too small to be a real model file. "
            f"This is the classic symptom of a Git LFS pointer file instead of the actual binary "
            f"(GitHub silently does this for files over ~100MB pushed without LFS configured properly). "
            f"First 200 bytes of the file:\n{head}"
        )

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

    try:
        model = load_model()
    except Exception as e:
        st.error("Model failed to load. Full details below (this is the real error Streamlit normally hides):")
        st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        st.stop()

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
