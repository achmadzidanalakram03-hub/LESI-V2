import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Konfigurasi Halaman (Harus di awal)
st.set_page_config(page_title="Deteksi Lesi Oral AI", layout="centered")

# Judul dan Deskripsi
st.title("🦷 Sistem Deteksi Lesi Rongga Mulut")
st.write("Sistem inferensi YOLOv11 untuk deteksi otomatis.")

# Fungsi memuat model dan di-cache agar tidak terus-menerus memuat (loading)
@st.cache_resource
def load_model():
    # Memanggil file best.pt dari repositori yang sama
    return YOLO('best.pt')

# Eksekusi pemanggilan model
model = load_model()

# Membuat modul unggah gambar
uploaded_file = st.file_uploader("Unggah foto klinis rongga mulut (Format: JPG/PNG)", type=["jpg", "jpeg", "png"])

# Logika jika pengguna sudah mengunggah foto
if uploaded_file is not None:
    # Menampilkan gambar yang diunggah dalam ukuran wajar
    image = Image.open(uploaded_file)
    
    # Membuat 2 kolom agar gambar tampil berdampingan (sebelum vs sesudah)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Citra Klinis Asli**")
        st.image(image, use_column_width=True)
        
    # Tombol untuk memicu AI
    if st.button('Jalankan Deteksi YOLO', type="primary"):
        with st.spinner('AI sedang memproses gambar...'):
            
            # Menjalankan deteksi AI
            results = model(image)
            
            # Mengekstraksi gambar hasil plot bounding box
            res_plotted = results[0].plot()
            
            with col2:
                st.markdown("**Hasil Deteksi Lesi**")
                st.image(res_plotted, use_column_width=True)
            
            st.success("Deteksi Selesai!")
