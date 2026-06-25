# 🩸 Babesia Infection Detector — Streamlit App

CNN + SoftMax classifier for automated Babesia infection detection from blood smear images.  
**SIMATS Engineering | MLA0409 Deep Learning for Applications**

---

## Files in this repo

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app |
| `requirements.txt` | Python dependencies |
| `Babesia_CNN_SoftMax_Colab.ipynb` | Training notebook (run in Google Colab) |
| `generate_model_locally.py` | Creates a dummy model for local UI testing only |
| `babesia_cnn_softmax_model.keras` | **Trained model — you must generate this yourself** (see below) |

---

## ⚠️ You need to generate the model file first

The `.keras` model file is NOT included here because it must be trained. Follow these steps:

### Step 1 — Train the model in Google Colab
1. Open `Babesia_CNN_SoftMax_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom (~10–15 minutes)
4. Cell 27 will automatically **save and download** `babesia_cnn_softmax_model.keras` to your computer

### Step 2 — Deploy on Streamlit Community Cloud (free)
1. Create a public GitHub repo (e.g. `babesia-detector-app`)
2. Push these files to the root of the repo:
   - `app.py`
   - `requirements.txt`
   - `babesia_cnn_softmax_model.keras`
3. If the `.keras` file is **larger than 100 MB**, use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git add babesia_cnn_softmax_model.keras
   git commit -m "Add model via LFS"
   git push
   ```
4. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**
5. Select your repo, branch `main`, file `app.py` → **Deploy**
6. Wait 2–3 minutes → you get a live public URL 🎉

---

## Run locally (for testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Option A: Use the real trained model from Colab (place it here first)
streamlit run app.py

# Option B: Quick UI test with a dummy (untrained) model
python generate_model_locally.py   # creates babesia_cnn_softmax_model.keras
streamlit run app.py
```

---

## About

Babesia parasites are visually almost identical to malaria parasites under the microscope (both are intra-erythrocytic protozoa producing ring-shaped inclusions in red blood cells). Since no large public labelled Babesia image dataset exists, this project trains on the NIH/Kaggle Malaria Blood Smear dataset as an academically accepted proxy.  

> This is a **proof-of-concept academic tool**, not a validated clinical diagnostic device.
