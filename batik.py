import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Memuat model .keras
model_path = "best_model_1.keras"  # Ganti dengan path model Anda
model = tf.keras.models.load_model(model_path)

# Nama kelas dari model (misal, motif batik)
class_labels = [
    "Insang",
    "Kawung",
    "Megamendung",
    "Parang",
    "Sidoluhur",
    "Truntum",
    "Tumpal",
]


# Fungsi untuk memproses gambar yang diunggah untuk prediksi
def preprocess_image(img_file):
    img = image.load_img(
        img_file, target_size=(224, 224)
    )  # Ubah ukuran gambar sesuai input model
    img_array = image.img_to_array(img)  # Mengonversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )  # Preprocessing sesuai dengan model
    return img_array


# Fungsi untuk memprediksi gambar
def predict_image(img_file):
    # Proses gambar
    img_array = preprocess_image(img_file)

    # Prediksi menggunakan model
    predictions = model.predict(img_array)

    # Menentukan kelas yang diprediksi dan kepercayaan (confidence)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions, axis=1)[0] * 100

    return predicted_class_label, confidence


# Menggunakan Streamlit untuk upload gambar
st.title("Aplikasi Klasifikasi Motif Batik")

uploaded_file = st.file_uploader(
    "Pilih Gambar Batik untuk Klasifikasi", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah hanya sekali
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True, width=300)

    # Melakukan prediksi dan menampilkan hasil hanya sekali
    predicted_class_label, confidence = predict_image(uploaded_file)

    # Menampilkan prediksi yang telah dilakukan
    st.write(f"**Nama Motif Batik**: {predicted_class_label}")
    st.write(f"**Kepercayaan**: {confidence:.2f}%")

else:
    st.write("Silakan unggah gambar untuk diklasifikasikan.")
