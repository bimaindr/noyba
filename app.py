import os
# WAJIB: Baris pertama untuk memastikan kompatibilitas Keras 2/3
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =========================
# CONFIG & LOAD MODEL
# =========================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")

@st.cache_resource
def load_model():
    # Model yang dibuat di Colab sering butuh safe_mode=False saat di-load di environment lain
    return tf.keras.models.load_model(
        "model_parurasio801010.h5", 
        compile=False, 
        safe_mode=False
    )

st.title("Deteksi Penyakit Paru-Paru (X-ray)")

try:
    model = load_model()
    # Hapus baris success jika sudah berjalan lancar agar UI bersih
    st.sidebar.success("Model Loaded Successfully")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================
# UI & PREDICTION
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # Preprocessing (Pastikan ukuran 224x224 sesuai training Anda)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Menganalisis..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.divider()
    st.success(f"Hasil Prediksi: **{class_names[predicted_class].upper()}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

    # Grafik Probabilitas
    prob_df = pd.DataFrame({"Probabilitas": prediction[0]}, index=class_names)
    st.bar_chart(prob_df)
