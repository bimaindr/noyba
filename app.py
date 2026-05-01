import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import sys

# ==========================================================
# CONFIG PAGE & UI
# ==========================================================
st.set_page_config(page_title="Deteksi Paru-Paru", layout="centered")

st.title("Deteksi Penyakit Paru-Paru (X-ray)")
st.write("Upload gambar X-ray thorax untuk analisis penyakit.")

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = "model_parurasio801010.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================================================
# KONFIGURASI LABEL & PREPROCESS
# ==========================================================
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    # Normalisasi
    image = image / 255.0

    # Tambah batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# ==========================================================
# UPLOAD & PREDIKSI
# ==========================================================
uploaded_file = st.file_uploader(
    "Pilih gambar X-ray (JPG/PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    img_array = preprocess_image(image)

    with st.spinner("Sedang menganalisis gambar..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.divider()
    st.subheader("Hasil Analisis")
    st.success(f"Hasil Prediksi: **{class_names[predicted_class].upper()}**")
    st.info(f"Tingkat Keyakinan: **{confidence*100:.2f}%**")

    # Visualisasi probabilitas
    prob_df = pd.DataFrame({
        "Probabilitas": prediction[0]
    }, index=class_names)

    st.bar_chart(prob_df)

# ==========================================================
# DEBUG INFO
# ==========================================================
with st.expander("Detail Sistem"):
    st.write("Python:", sys.version)
    st.write("TensorFlow:", tf.__version__)
    st.write("Keras:", tf.keras.__version__)
