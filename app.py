# Import libraries
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
status_univ_mapping = {"PTN": 1, "PTS Bojonegoro": 2, "PTS Luar Bojonegoro": 3}
jenjang_mapping = {"S1": 1, "D4": 2, "D3": 3}
akreditasi_mapping = {"A": 1, "B": 2, "C": 3}
kartu_mapping = {"Ya": 1, "Tidak": 0}

status_univ = st.selectbox("Status Universitas", list(status_univ_mapping.keys()))
jenjang = st.selectbox("Jenjang Mahasiswa", list(jenjang_mapping.keys()))
akreditasi = st.selectbox("Akreditasi Program Studi", list(akreditasi_mapping.keys()))
kartu = st.selectbox("Memiliki Kartu Mahasiswa", list(kartu_mapping.keys()))

# Konversi ke nilai numerik
status_univ_numeric = status_univ_mapping[status_univ]
jenjang_numeric = jenjang_mapping[jenjang]
akreditasi_numeric = akreditasi_mapping[akreditasi]
kartu_numeric = kartu_mapping[kartu]

# Tombol untuk memprediksi
if st.button("Prediksi"):
    # Mengumpulkan data untuk prediksi
    input_data = pd.DataFrame({
        'Status_Univ': [status_univ_numeric],
        'Jenjang': [jenjang_numeric],
        'Akreditasi': [akreditasi_numeric],
        'Kartu': [kartu_numeric]
    })

    # Handle missing values by filling with the mean
    input_data_filled = input_data.apply(pd.to_numeric, errors='coerce').fillna(input_data.mean())

    # One-hot encode categorical variables
    input_data_encoded = pd.get_dummies(input_data_filled)

    # Ensure input_data_encoded has the same columns as during training
    # Use reindex to add missing columns (if any) with default values
    input_data_encoded = input_data_encoded.reindex(columns=knn_model['columns'], fill_value=0)

    # Print input_data_encoded for debugging
    st.write("Input Data Encoded:", input_data_encoded)

    # Melakukan prediksi
    hasil_prediksi = predict_beasiswa(input_data_encoded)

    # Menampilkan hasil prediksi
    if hasil_prediksi[0] == 1:
        st.success(f"Mahasiswa dari {status_univ} dengan jenjang {jenjang}, akreditasi {akreditasi}, dan {kartu} kartu layak untuk menerima beasiswa!")
    else:
        st.error(f"Mahasiswa dari {status_univ} dengan jenjang {jenjang}, akreditasi {akreditasi}, dan {kartu} kartu tidak layak untuk menerima beasiswa.")
