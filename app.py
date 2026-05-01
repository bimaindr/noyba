import os
# WAJIB: Aktifkan mode legacy agar TensorFlow bisa membaca model .h5 lama dengan benar
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
# Nama file harus sama persis dengan yang ada di repository GitHub kamu
MODEL_PATH = "model_parurasio801010.h5" 

@st.cache_resource
def load_model():
    # Menggunakan compile=False karena kita hanya butuh untuk prediksi (inference)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# Penanganan error jika model gagal dimuat
try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan file model .h5 sudah di-upload ke GitHub dan versi library sudah sesuai.")
    st.stop()

# =========================
# LABEL KELAS
# =========================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Konversi ke RGB untuk memastikan 3 channel (mencegah error pada gambar grayscale)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    # Gunakan ukuran 224x224 sesuai arsitektur EfficientNet/CNN yang umum digunakan
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDIKSI
    # =========================
    with st.spinner("Menganalisis gambar..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # =========================
    # OUTPUT HASIL
    # =========================
    st.divider()
    st.subheader("Hasil Analisis")
    
    # Menampilkan hasil dengan teks tebal dan warna sukses
    st.success(f"Hasil Prediksi: **{class_names[predicted_class].upper()}**")
    st.info(f"Tingkat Keyakinan (Confidence): **{confidence*100:.2f}%**")

    # Tampilkan grafik probabilitas agar lebih informatif
    st.subheader("Probabilitas Tiap Kelas")
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)
    st.bar_chart(prob_df)