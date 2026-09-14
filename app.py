import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# Konfigurasi Halaman (Mode Wide untuk gaya Dashboard)
st.set_page_config(page_title="AI Lesion Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- MEMUAT MODEL AI ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("🩺 AI Med-Dashboard")
    st.markdown("---")
    menu = st.radio(
        "Menu Utama",
        ["Dashboard", "Riwayat Deteksi", "Feature", "About", "Project", "Contact"]
    )
    st.markdown("---")
    st.caption("Masuk sebagai: Admin Klinis")

# --- KONTEN UTAMA ---
if menu == "Dashboard":
    st.title("Dashboard Analisis Lesi Rongga Mulut")
    st.markdown("Ringkasan performa AI dan antarmuka deteksi langsung.")
    
    # Bagian 1: Grafik Kinerja & Metrik Real-time (Simulasi)
    st.markdown("### Kinerja Model AI")
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    col_metric1.metric("Akurasi (mAP50)", "94.5%", "+1.2%")
    col_metric2.metric("Total Pasien Diperiksa", "1,204", "+15 Hari ini")
    col_metric3.metric("Waktu Inferensi Rata-rata", "0.8s", "-0.2s")
    col_metric4.metric("Status Server", "Online", "Stabil")

    # Grafik simulasi distribusi lesi harian menggunakan data acak
    st.markdown("#### Tren Deteksi Harian (Geo-tongue, Candidiasis, BHT)")
    chart_data = pd.DataFrame(
        np.random.randint(1, 10, size=(10, 3)),
        columns=['Geo-tongue', 'Candidiasis', 'BHT']
    )
    st.line_chart(chart_data, use_container_width=True)
    st.markdown("---")

    # Bagian 2: Antarmuka Deteksi Model AI
    st.markdown("### Modul Deteksi AI")
    uploaded_file = st.file_uploader("Unggah Citra Klinis Pasien (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**Citra Masukan**")
            st.image(image, use_container_width=True)
            
        if st.button('Jalankan Analisis YOLO', type="primary", use_container_width=True):
            with st.spinner('Menghitung *bounding box* dan probabilitas...'):
                results = model(image)
                res_plotted = results[0].plot()
                
                with col_img2:
                    st.markdown("**Hasil Identifikasi AI**")
                    st.image(res_plotted, use_container_width=True)

# --- HALAMAN PLACEHOLDER LAINNYA ---
elif menu == "Riwayat Deteksi":
    st.title("Riwayat Deteksi")
    st.write("Tabel *database* pasien dan log hasil klasifikasi model akan tampil di sini.")
    
elif menu == "Feature":
    st.title("Fitur Sistem")
    st.write("Penjelasan kapabilitas arsitektur YOLOv11 dalam menganalisis citra oral.")

elif menu == "About":
    st.title("Tentang Sistem")
    st.write("Aplikasi skrining lesi mulut berbasis *Deep Learning* untuk kolaborasi interprofesional.")

elif menu == "Project":
    st.title("Project Overview")
    st.write("Dokumentasi metrik, *confusion matrix*, dan distribusi *dataset* pelatihan.")

elif menu == "Contact":
    st.title("Hubungi Kami")
    st.write("Informasi kontak pengembang medis dan teknis.")
