# Import library
import streamlit as st
import pandas as pd
import joblib

# Load KNN model
knn_model = joblib.load('beasiswa.pkl')

# Fungsi untuk prediksi beasiswa
def predict_beasiswa(data):
    prediction = knn_model.predict(data)
    return prediction

# Judul halaman web
st.title("Aplikasi Prediksi Penerima Beasiswa")

# Input data mahasiswa
status_univ = st.selectbox("Status Universitas", ["PTN", "PTS Bojonegoro", "PTS Luar Bojonegoro"])
jenjang = st.selectbox("Jenjang Mahasiswa", ["S1", "D4", "D3"])
akreditasi = st.selectbox("Akreditasi Program Studi", ["A", "B", "C"])
kartu = st.selectbox("Memiliki Kartu Mahasiswa", ["Ya", "Tidak"])

# Tombol untuk memprediksi
if st.button("Prediksi"):
    # Mengumpulkan data untuk prediksi
    input_data = pd.DataFrame({
        'Status_Univ': [status_univ],
        'Jenjang': [jenjang],
        'Akreditasi': [akreditasi],
        'Kartu': [kartu]
    })

    # Ensure input data has the same columns and order as during training
    input_data = input_data[['Status_Univ', 'Jenjang', 'Akreditasi', 'Kartu']]

    # Melakukan prediksi
    hasil_prediksi = predict_beasiswa(input_data)

    # Menampilkan hasil prediksi
    if hasil_prediksi[0] == 1:
        st.success(f"Mahasiswa dari {status_univ} dengan jenjang {jenjang}, akreditasi {akreditasi}, dan {kartu} kartu layak untuk menerima beasiswa!")
    else:
        st.error(f"Mahasiswa dari {status_univ} dengan jenjang {jenjang}, akreditasi {akreditasi}, dan {kartu} kartu tidak layak untuk menerima beasiswa.")
