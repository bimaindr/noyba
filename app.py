import os
# WAJIB: Letakkan di baris pertama untuk memaksa penggunaan Keras legacy (lama)
# Ini krusial untuk menangani error deserialisasi pada layer 'InputLayer'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ==========================================================
# CONFIG PAGE & UI
# ==========================================================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")

st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray untuk mendapatkan prediksi penyakit.")

# ==========================================================
# LOAD MODEL
# ==========================================================
# Nama file disesuaikan dengan struktur folder: model_parurasio801010.h5
MODEL_PATH = "model_parurasio801010.h5" 

@st.cache_resource
def load_model():
    # safe_mode=False digunakan untuk mengatasi error 'Unrecognized keyword arguments'
    # compile=False digunakan karena kita hanya melakukan inferensi (prediksi)
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Tips: Pastikan runtime.txt diatur ke python-3.10.11 dan hapus aplikasi di dashboard Streamlit untuk Deploy ulang.")
    st.stop()

# ==========================================================
# KONFIGURASI LABEL & PREPROCESS
# ==========================================================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]
IMG_SIZE = 224

def preprocess_image(image):
    # Resize sesuai arsitektur training
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    
    # Normalisasi
    image = image / 255.0
    
    # Menambah dimensi batch (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    return image

# ==========================================================
# LOGIKA UPLOAD & PREDIKSI
# ==========================================================
uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Konversi ke RGB untuk menjaga konsistensi channel (3 channel)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # Jalankan preprocessing
    img_array = preprocess_image(image)

    with st.spinner("Menganalisis gambar..."):
        # Melakukan prediksi
        prediction = model.predict(img_array)

    # Ambil indeks kelas tertinggi dan nilai konfidensinya
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Tampilkan hasil
    st.divider()
    st.subheader("Hasil Analisis")
    st.success(f"Hasil Prediksi: **{class_names[predicted_class].upper()}**")
    st.info(f"Tingkat Keyakinan (Confidence): **{confidence*100:.2f}%**")

    # Visualisasi Probabilitas
    st.subheader("Probabilitas Tiap Kelas")
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)
    st.bar_chart(prob_df)

# ==========================================================
# FOOTER / DEBUG INFO
# ==========================================================
with st.expander("Informasi Sistem"):
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Model Path: {MODEL_PATH}")
