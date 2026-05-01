import os
# WAJIB: Baris pertama sebelum import tensorflow untuk memaksa mesin Keras lama
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
st.write("Upload gambar X-ray thorax untuk analisis penyakit.")

# ==========================================================
# LOAD MODEL
# ==========================================================
# Nama file harus sesuai dengan file di GitHub Anda
MODEL_PATH = "model_parurasio801010.h5" 

@st.cache_resource
def load_model():
    # safe_mode=False untuk mengatasi error 'Unrecognized keyword arguments'
    # compile=False karena model hanya untuk prediksi (inference)
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan runtime.txt berisi python-3.10.11 dan lakukan 'Delete App' lalu deploy ulang.")
    st.stop()

# ==========================================================
# KONFIGURASI LABEL & PREPROCESS
# ==========================================================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]
IMG_SIZE = 224

def preprocess_image(image):
    # Resize gambar ke ukuran 224x224 (standar Colab/EfficientNet)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    
    # Normalisasi piksel ke rentang 0-1
    image = image / 255.0
    
    # Menambah dimensi batch (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)
    return image

# ==========================================================
# LOGIKA UPLOAD & PREDIKSI
# ==========================================================
uploaded_file = st.file_uploader("Pilih gambar X-ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Konversi ke RGB untuk memastikan 3 channel warna
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # Proses preprocessing
    img_array = preprocess_image(image)

    with st.spinner("Sedang menganalisis gambar..."):
        # Melakukan prediksi
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Menampilkan hasil
    st.divider()
    st.subheader("Hasil Analisis")
    st.success(f"Hasil Prediksi: **{class_names[predicted_class].upper()}**")
    st.info(f"Tingkat Keyakinan (Confidence): **{confidence*100:.2f}%**")

    # Visualisasi probabilitas per kelas
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)
    st.bar_chart(prob_df)

# ==========================================================
# FOOTER / DEBUG INFO
# ==========================================================
with st.expander("Detail Sistem"):
    st.write(f"TensorFlow Version: {tf.__version__}")
    st.write(f"Keras Backend: {tf.keras.__name__}")
