import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# --- KUSTOMISASI CSS (UI/UX MODERN) ---
st.markdown("""
    <style>
    /* Latar belakang utama aplikasi */
    .stApp {
        background-color: #F4F7F6;
    }
    
    /* Styling untuk Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* Kartu Metrik (KPI) bergaya Dribbble */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2A9D8F;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #264653;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Kontainer Gambar Deteksi */
    .image-container {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
    }
    
    /* Tombol Utama */
    .stButton>button {
        background-color: #2A9D8F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21867a;
        box-shadow: 0 4px 8px rgba(42, 157, 143, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI DATABASE CSV ---
DB_FILE = "log_deteksi.csv"

def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["Waktu", "Tanggal", "Lesi_Terdeteksi", "Confidence"])
        df.to_csv(DB_FILE, index=False)

init_db()

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60) # Ikon medis simpel
    st.markdown("### RSGM Unjani")
    st.caption("AI Dental Vision System")
    st.markdown("---")
    
    menu = st.radio(
        "Navigasi",
        ["Dashboard", "Riwayat Deteksi", "Feature", "About", "Project", "Contact"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**User Profile**")
    st.markdown("👨‍⚕️ drg. Adinara Savero, S.KG")
    st.caption("Status: Clinical Clerkship (Aktif)")

# --- KONTEN UTAMA ---
if menu == "Dashboard":
    # Header Section
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.title("Sistem Skrining Lesi Oral")
        st.markdown(f"<p style='color: #6c757d;'>Tanggal Hari Ini: {datetime.now().strftime('%d %B %Y')}</p>", unsafe_allow_html=True)
    
    # 1. TARIK DATA AKTUAL (QUERY)
    df_log = pd.read_csv(DB_FILE)
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0%"

    # Custom HTML KPI Cards
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Pemeriksaan AI</p>
            <p class="metric-value">{total_deteksi}</p>
            <p style="color: #2A9D8F; font-size: 0.8rem; margin:0;">▲ {deteksi_hari_ini} pasien hari ini</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Rata-rata Confidence</p>
            <p class="metric-value">{avg_conf}</p>
            <p style="color: #E9C46A; font-size: 0.8rem; margin:0;">Berdasarkan YOLOv11</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Status Integrasi</p>
            <p class="metric-value">Sinkron</p>
            <p style="color: #2A9D8F; font-size: 0.8rem; margin:0;">Database Real-time Aktif</p>
        </div>
        """, unsafe_allow_html=True)

    # Grafik Area Real-time
    st.markdown("<h4 style='color: #264653; margin-top: 20px;'>Grafik Distribusi Lesi</h4>", unsafe_allow_html=True)
    if total_deteksi > 0:
        chart_data = df_log.groupby(['Tanggal', 'Lesi_Terdeteksi']).size().unstack(fill_value=0)
        st.area_chart(chart_data, use_container_width=True)
    else:
        st.info("Menunggu data deteksi pertama masuk ke dalam sistem.")
    
    st.markdown("---")

    # 2. ANTARMUKA DETEKSI AI
    st.markdown("<h4 style='color: #264653;'>Modul Analisis Citra Klinis</h4>", unsafe_allow_html=True)
    
    # Area Unggah diubah menjadi lebih compact
    uploaded_file = st.file_uploader("Seret dan lepas (Drag & Drop) foto intraoral pasien di sini", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_img1, col_img2 = st.columns(2, gap="large")
        
        with col_img1:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.markdown("**📸 Citra Klinis Masukan**")
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button('Mulai Analisis YOLO', use_container_width=True)
            
        if analyze_btn:
            with st.spinner('Memindai anomali dental...'):
                results = model(image)
                res_plotted = results[0].plot()
                
                # SUNTIKKAN PERINTAH SIMPAN
                boxes = results[0].boxes
                if len(boxes) > 0:
                    new_records = []
                    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tanggal_sekarang = datetime.now().strftime("%Y-%m-%d")
                    
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        conf_score = float(box.conf[0].item())
                        nama_lesi = model.names[class_id] 
                        
                        new_records.append({
                            "Waktu": waktu_sekarang,
                            "Tanggal": tanggal_sekarang,
                            "Lesi_Terdeteksi": nama_lesi,
                            "Confidence": round(conf_score, 2)
                        })
                    
                    df_new = pd.DataFrame(new_records)
                    df_new.to_csv(DB_FILE, mode='a', header=False, index=False)
                
                with col_img2:
                    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                    st.markdown("**🧬 Hasil Pemetaan Bounding Box**")
                    st.image(res_plotted, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if len(boxes) == 0:
                        st.warning("Jaringan sehat / Tidak ada lesi yang terdeteksi secara spesifik.")
                    else:
                        st.success("✅ Log pasien berhasil diperbarui ke database.")

elif menu == "Riwayat Deteksi":
    st.title("Riwayat Data Pasien")
    df_log = pd.read_csv(DB_FILE)
    st.dataframe(df_log, use_container_width=True)
    
    with open(DB_FILE, "rb") as file:
        st.download_button(
            label="Unduh Laporan CSV",
            data=file,
            file_name="Laporan_Deteksi_Lesi.csv",
            mime="text/csv"
        )
        
elif menu == "Feature":
    st.title("Fitur Sistem")
    st.write("Sistem inferensi ini didukung oleh arsitektur YOLOv11 yang dioptimalkan untuk mendeteksi *bounding box* anomali oral. Pemantauan metrik dan distribusi gambar langsung disinkronkan ke dalam *dashboard*.")

elif menu == "About":
    st.title("Tentang Aplikasi")
    st.write("Aplikasi skrining ini dirancang untuk memfasilitasi pengambilan keputusan klinis dan mendukung kolaborasi interprofesional di lingkungan layanan kesehatan tingkat pertama maupun rumah sakit pendidikan.")

elif menu == "Project":
    st.title("Project Overview")
    st.write("Area ini didedikasikan untuk menampilkan visualisasi *confusion matrix*, kurva presisi-recall, dan performa uji hipotesis dari iterasi model pelatihan.")

elif menu == "Contact":
    st.title("Hubungi Pengembang")
    st.write("Untuk kebutuhan teknis dan kalibrasi model, silakan hubungi tim administrator klinis.")
