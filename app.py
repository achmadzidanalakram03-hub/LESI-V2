import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# --- KUSTOMISASI CSS (RESPONSIF & PALET DRIBBBLE) ---
st.markdown("""
    <style>
    /* Latar belakang utama aplikasi */
    .stApp {
        background-color: #FAFAFD;
    }
    
    /* Styling untuk Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(18, 32, 86, 0.05);
    }
    
    /* Mengubah warna teks utama */
    .stMarkdown, .stText, h1, h2, h3, h4 {
        color: #122056 !important;
    }
    
    /* Kartu Metrik (KPI) Desktop */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(18, 32, 86, 0.05);
        border-left: 5px solid #5B65DC;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #122056;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #122056;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-sub {
        color: #5B65DC;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0;
        margin-top: 5px;
    }
    
    /* Kontainer Gambar */
    .image-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(18, 32, 86, 0.05);
        border: 1px solid #EEEFFD;
        margin-bottom: 15px;
    }
    
    /* Tombol Utama Desktop */
    .stButton>button {
        background-color: #5B65DC;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #122056;
        box-shadow: 0 4px 12px rgba(91, 101, 220, 0.4);
    }
    
    /* === ATURAN KHUSUS LAYAR MOBILE (HP/TABLET KECIL) === */
    @media (max-width: 768px) {
        .metric-card {
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #5B65DC;
        }
        .metric-value {
            font-size: 1.8rem; /* Teks angka dikecilkan di HP */
        }
        .metric-label {
            font-size: 0.8rem;
        }
        .image-container {
            padding: 10px; /* Jarak bingkai dirapatkan */
        }
        h1 {
            font-size: 1.8rem !important; /* Judul disesuaikan */
        }
        .stButton>button {
            padding: 15px; /* Tombol dibuat lebih tinggi agar mudah disentuh jari (Touch Target) */
            font-size: 1.1rem;
        }
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
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.markdown("<h3 style='margin-bottom:0;'>RSGM Unjani</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5B65DC; font-size:0.8rem; font-weight:bold;'>AI Dental Vision System</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Navigasi",
        ["Dashboard", "Riwayat Deteksi", "Feature", "About", "Project", "Contact"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<strong>User Profile</strong>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0;'>👨‍⚕️ drg. Adinara Savero, S.KG</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5B65DC; font-size:0.8rem;'>Status: Clinical Clerkship (Aktif)</p>", unsafe_allow_html=True)

# --- KONTEN UTAMA ---
if menu == "Dashboard":
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("<h1>Sistem Skrining Lesi Oral</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='opacity: 0.7; margin-top:-10px;'>Tanggal Hari Ini: {datetime.now().strftime('%d %B %Y')}</p>", unsafe_allow_html=True)
    
    df_log = pd.read_csv(DB_FILE)
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0%"

    st.markdown("<br>", unsafe_allow_html=True)
    # Di desktop tampil 3 kolom, di mobile otomatis menumpuk jadi 1 kolom berbaris bawah
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Pemeriksaan AI</p>
            <p class="metric-value">{total_deteksi}</p>
            <p class="metric-sub">▲ {deteksi_hari_ini} pasien hari ini</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Rata-rata Confidence</p>
            <p class="metric-value">{avg_conf}</p>
            <p class="metric-sub">Berdasarkan YOLOv11</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Status Integrasi</p>
            <p class="metric-value" style="color:#5B65DC;">Sinkron</p>
            <p class="metric-sub">Database Real-time Aktif</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h4 style='margin-top: 20px;'>Grafik Distribusi Lesi</h4>", unsafe_allow_html=True)
    if total_deteksi > 0:
        chart_data = df_log.groupby(['Tanggal', 'Lesi_Terdeteksi']).size().unstack(fill_value=0)
        st.area_chart(chart_data, use_container_width=True)
    else:
        st.info("Menunggu data deteksi pertama masuk ke dalam sistem.")
    
    st.markdown("---")

    st.markdown("<h4>Modul Analisis Citra Klinis</h4>", unsafe_allow_html=True)
    
    # Menambahkan antarmuka Tab untuk memilih antara Galeri atau Kamera Langsung (Depan/Belakang)
    tab_unggah, tab_kamera = st.tabs(["Upload", "Kamera"])
    
    image = None # Variabel penampung citra
    
    with tab_unggah:
        uploaded_file = st.file_uploader("Pilih foto intraoral dari penyimpanan perangkat", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
    with tab_kamera:
        camera_file = st.camera_input("Ambil gambar lesi secara langsung (Gunakan ikon rotate kamera bawaan HP untuk opsi depan/belakang)")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')

    # Mengeksekusi penanganan citra jika gambar sudah tersedia dari salah satu sumber
    if image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        # gap="large" memberikan ruang nafas antar kolom di desktop, dan jarak vertikal saat ditumpuk di mobile
        col_img1, col_img2 = st.columns(2, gap="large")
        
        with col_img1:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.markdown("<strong>📸 Citra Klinis Masukan</strong>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button('Mulai Analisis YOLO', use_container_width=True)
            
        if analyze_btn:
            with st.spinner('Memindai anomali dental...'):
                results = model(image)
                res_plotted = results[0].plot()
                
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
                    st.markdown("<strong>🧬 Hasil Pemetaan Bounding Box</strong>", unsafe_allow_html=True)
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
        st.download_button("Unduh Laporan CSV", data=file, file_name="Laporan_Deteksi_Lesi.csv", mime="text/csv")
        
elif menu == "Feature":
    st.title("Fitur Sistem")
    st.write("Sistem inferensi didukung arsitektur YOLOv11. Pemantauan metrik dan distribusi disinkronkan ke dalam dashboard.")

elif menu == "About":
    st.title("Tentang Aplikasi")
    st.write("Aplikasi skrining dirancang memfasilitasi keputusan klinis dan kolaborasi interprofesional.")

elif menu == "Project":
    st.title("Project Overview")
    st.write("Dokumentasi matriks kebingungan dan performa uji hipotesis.")

elif menu == "Contact":
    st.title("Hubungi Pengembang")
    st.write("Untuk kalibrasi model, hubungi tim administrator.")
