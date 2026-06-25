"""
Streamlit web app for the Babesia (CNN + SoftMax) blood-smear detector.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push app.py + requirements.txt + babesia_cnn_softmax_model.keras to a public GitHub repo.
    2. Go to https://share.streamlit.io → New app → select repo → Deploy.

    ⚠️  If the .keras file is larger than 100 MB, use Git LFS:
        git lfs install
        git lfs track "*.keras"
        git add .gitattributes babesia_cnn_softmax_model.keras
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "babesia_cnn_softmax_model.keras"
IMG_SIZE   = 128
CLASS_NAMES = ["Normal", "Infected"]

st.set_page_config(
    page_title="Babesia Infection Detector",
    page_icon="🩸",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"**Model file not found:** `{MODEL_PATH}`\n\n"
            "Please ensure `babesia_cnn_softmax_model.keras` is in the same folder as `app.py`.\n\n"
            "**How to get it:**\n"
            "1. Open `Babesia_CNN_SoftMax_Colab.ipynb` in Google Colab (Runtime → T4 GPU).\n"
            "2. Run all cells — training takes ~10-15 minutes.\n"
            "3. Cell 27 saves and auto-downloads `babesia_cnn_softmax_model.keras`.\n"
            "4. Place that file next to `app.py` and restart the app."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🩸 Automated Babesia Infection Detection")
    st.caption("CNN + SoftMax Classifier — Blood Smear Image Analysis")
    st.caption("SIMATS Engineering | MLA0409 Deep Learning for Applications")

    st.warning(
        "⚠️  This model was trained on the public NIH/Kaggle malaria blood-smear dataset as a "
        "research proxy for Babesia, due to the documented visual similarity between the two "
        "intra-erythrocytic parasites and the lack of a public Babesia image bank. "
        "It is a **proof-of-concept academic tool — not a validated clinical diagnostic device**. "
        "Do not use it to make real medical decisions."
    )

    model = load_model()

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload a blood smear cell image (.png / .jpg / .jpeg)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col_res:
            with st.spinner("Analysing image…"):
                batch  = preprocess_image(image)
                probs  = model.predict(batch, verbose=0)[0]
                pred_idx   = int(np.argmax(probs))
                pred_label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx]) * 100

            st.subheader("Result")
            if pred_label == "Infected":
                st.error(f"🔴 **{pred_label}**")
            else:
                st.success(f"🟢 **{pred_label}**")

            st.metric("Confidence", f"{confidence:.2f}%")

            st.markdown("**Class probabilities**")
            col1, col2 = st.columns(2)
            col1.metric("P(Normal)",   f"{probs[0] * 100:.2f}%")
            col2.metric("P(Infected)", f"{probs[1] * 100:.2f}%")

            st.markdown("**Infection probability**")
            st.progress(float(probs[1]))
    else:
        st.info("⬆️  Upload a blood smear cell image above to get a prediction.")

    st.divider()
    with st.expander("ℹ️  About this project"):
        st.markdown(
            """
            **Project:** Automated Babesia Infection Detection from Blood Sample Images  
            **Method:** Custom CNN feature extractor → SoftMax classification head  
            **Dataset:** NIH/Kaggle Malaria Blood Smear Dataset (used as Babesia proxy)  
            **Classes:** Normal (Uninfected) · Infected (Parasitized)  
            **Image size:** 128 × 128 px  
            **Framework:** TensorFlow / Keras  
            """
        )


if __name__ == "__main__":
    main()
