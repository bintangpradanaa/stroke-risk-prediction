import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("best_stroke_prediction_model.pkl")
scaler = joblib.load("scaler_stroke_model.pkl")

# Judul halaman di tengah
st.markdown("<h1 style='text-align: center;'>Prediksi Risiko Stroke</h1>", unsafe_allow_html=True)
st.markdown("Silakan isi data berikut untuk memprediksi kemungkinan risiko stroke:")

# Dua kolom input
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia (dalam tahun)", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Pernah/Sedang Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Pernah/Sedang Memiliki Penyakit Jantung", ["Tidak", "Ya"])
    ever_married = st.selectbox("Status Pernikahan", ["Belum Menikah", "Sudah Menikah"])

with col2:
    work_type = st.selectbox("Jenis Pekerjaan", ["Pegawai Swasta", "Wiraswasta", "Pegawai Negeri", "Anak-anak", "Belum Pernah Bekerja"])
    residence_type = st.selectbox("Tipe Tempat Tinggal", ["Perkotaan", "Pedesaan"])
    avg_glucose_level = st.number_input("Rata-rata Glukosa Darah", min_value=0.0, step=0.1, format="%.2f")
    bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, step=0.1, format="%.2f")
    smoking_status = st.selectbox("Status Merokok", ["Pernah merokok", "Tidak Pernah", "Masih Merokok"])

# Mapping input ke format numerik (sesuai pelatihan model)
gender_map = {"Laki-laki": 1, "Perempuan": 0}
yes_no_map = {"Ya": 1, "Tidak": 0}
married_map = {"Sudah Menikah": 1, "Belum Menikah": 0}
residence_map = {"Perkotaan": 1, "Pedesaan": 0}
smoking_map = {
    "Tidak Pernah": 0,
    "Pernah merokok": 1,
    "Masih Merokok": 2
}
work_type_map = {
    "Pegawai Swasta": 0,
    "Wiraswasta": 1,
    "Pegawai Negeri": 2,
    "Anak-anak": 3,
    "Belum Pernah Bekerja": 4
}

# Tombol prediksi
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Prediksi Risiko Stroke", use_container_width=True):
    input_data = pd.DataFrame([[
        gender_map[gender],
        age,
        yes_no_map[hypertension],
        yes_no_map[heart_disease],
        married_map[ever_married],
        work_type_map[work_type],
        residence_map[residence_type],
        avg_glucose_level,
        bmi,
        smoking_map[smoking_status]
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])

    # Transformasi data
    input_scaled = scaler.transform(input_data)
    stroke_prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Hasil Prediksi")
    st.write(f"**Probabilitas Risiko Stroke:** {stroke_prob * 100:.2f}%")

    # Kategori risiko
    if stroke_prob < 0.25:
        st.success("✅ Risiko stroke sangat rendah.")
    elif stroke_prob < 0.5:
        st.info("🟦 Risiko stroke cukup rendah.")
    elif stroke_prob < 0.75:
        st.warning("⚠️ Risiko stroke cukup tinggi.")
    else:
        st.error("❌ Risiko stroke sangat tinggi.")
