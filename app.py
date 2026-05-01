import os
# WAJIB: Letakkan paling atas untuk memaksa penggunaan Keras lama
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")

st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi.")

# =========================
# LOAD MODEL
# =========================
# Pastikan nama file ini sesuai dengan yang ada di GitHub kamu
MODEL_PATH = "model_parurasio801010.h5" 

@st.cache_resource
def load_model():
    # safe_mode=False digunakan untuk mengatasi error 'Unrecognized keyword arguments' pada InputLayer
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan file model .h5 sudah ada di repo dan versi library sesuai.")
    st.stop()

# =========================
# LABEL KELAS
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

# =========================
# UPLOAD & PREDIKSI
# =========================
uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
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

    # Chart
    prob_df = pd.DataFrame({"Probabilitas": prediction[0]}, index=class_names)
    st.bar_chart(prob_df)
