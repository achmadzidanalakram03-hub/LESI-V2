"""
Klinik AI RSGM — Sistem Skrining Lesi Oral berbasis YOLO
=========================================================
Terintegrasi dengan estetika UI/UX Modern ala Tailwind (Canva)
"""

import io
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
APP_VERSION = "2.0 (Modern UI)"
CLINIC_NAME = "RSGM Unjani"
USER_NAME = "drg. Adinara Savero, S.KG"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi.csv")
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi",
    "Confidence", "Model_Version", "Nama_File",
]

# PALET WARNA (Diadaptasi dari kode Canva)
PRIMARY = "#2A9D8F"       # Teal
PRIMARY_DARK = "#21867a"  # Deep Teal
ACCENT = "#E9C46A"        # Yellow
DANGER = "#E76F51"        # Coral
DARK_TEXT = "#264653"     # Ink
BG_MIST = "#F4F7F7"       # Mist Background

CONF_HIGH = 0.75
CONF_MED = 0.50

LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum (Cheek Biting)",
        "deskripsi": "Lesi traumatik pada mukosa bukal akibat kebiasaan menggigit pipi berulang, umumnya tampak sebagai area putih ireguler.",
        "rekomendasi": "Edukasi pasien untuk menghentikan kebiasaan menggigit pipi; evaluasi ulang bila lesi menetap lebih dari 2 minggu.",
        "urgensi": "Rendah",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue (Lidah Berlapis)",
        "deskripsi": "Lapisan putih hingga kekuningan pada dorsum lidah akibat penumpukan debris, bakteri, dan sel epitel deskuamasi.",
        "rekomendasi": "Instruksikan pembersihan lidah rutin (tongue scraper) dan evaluasi kebersihan mulut secara umum.",
        "urgensi": "Rendah",
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Kerusakan jaringan keras gigi akibat proses demineralisasi oleh asam hasil metabolisme bakteri plak.",
        "rekomendasi": "Rujuk untuk pemeriksaan klinis dan radiografis lanjutan guna menentukan rencana restorasi.",
        "urgensi": "Sedang-Tinggi",
    },
    "linea alba": {
        "nama_klinis": "Linea Alba",
        "deskripsi": "Garis putih horizontal pada mukosa bukal sepanjang bidang oklusal, umumnya akibat tekanan/gesekan kronis dan bersifat jinak.",
        "rekomendasi": "Umumnya tidak memerlukan tatalaksana khusus; monitor bila terjadi perubahan ukuran atau warna.",
        "urgensi": "Rendah",
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran vena pada permukaan ventral lidah, umum ditemukan pada individu usia lanjut, bersifat jinak.",
        "rekomendasi": "Tidak memerlukan tatalaksana khusus kecuali disertai gejala lain; edukasi pasien mengenai sifat jinak lesi.",
        "urgensi": "Rendah",
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Deposit mineral (kalkulus) dan/atau pewarnaan ekstrinsik pada permukaan gigi akibat akumulasi plak dan faktor eksternal.",
        "rekomendasi": "Rekomendasikan scaling profesional dan evaluasi kebiasaan oral hygiene pasien.",
        "urgensi": "Sedang",
    },
    "torus": {
        "nama_klinis": "Torus (Mandibularis/Palatinus)",
        "deskripsi": "Eksostosis tulang jinak pada mandibula atau palatum, umumnya asimtomatik.",
        "rekomendasi": "Tidak memerlukan tindakan kecuali mengganggu fungsi (bicara, protesa) atau membesar signifikan.",
        "urgensi": "Rendah",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Lesi ulseratif pada mukosa oral akibat trauma mekanis, termal, atau kimiawi, biasanya sembuh spontan.",
        "rekomendasi": "Evaluasi ulang bila tidak sembuh dalam 10-14 hari untuk menyingkirkan diagnosis banding lain.",
        "urgensi": "Sedang",
    },
}


# --- KONFIGURASI HALAMAN & INJEKSI CSS MODERN (Terinspirasi Tailwind/Canva) ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    /* Reset & Variabel Global */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
    }}
    
    .stApp {{ background-color: {BG_MIST}; }}
    
    /* Styling Sidebar ala Canva (Gelap) */
    [data-testid="stSidebar"] {{
        background-color: {DARK_TEXT} !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }}
    /* Teks Sidebar menjadi putih/terang */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio label {{
        color: #E5EEEE !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{ color: #E5EEEE !important; }}
    
    /* Kartu UI (Card) */
    .metric-card {{
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 7px 24px rgba(38,70,83,.07);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(38,70,83,.1);
    }}
    .metric-value {{ font-size: 2.2rem; font-weight: 700; color: {DARK_TEXT}; margin: 10px 0 0 0; line-height: 1.2; }}
    .metric-label {{
        font-size: 0.95rem; color: #5E7075;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    
    /* Container Gambar */
    .image-container {{
        background-color: #FFFFFF;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(38,70,83,.05);
        border: 1px solid #E2E8F0;
    }}
    
    /* Desain Tombol Utama */
    .stButton>button {{
        background-color: {PRIMARY} !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 6px rgba(42,157,143,.2) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover {{
        background-color: {PRIMARY_DARK} !important;
        box-shadow: 0 6px 12px rgba(42,157,143,.3) !important;
    }}
    
    /* Badge Urgensi */
    .urg-low {{ background-color:#dff4ef; color:#157267; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }}
    .urg-med {{ background-color:#fff3d6; color:#956a09; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }}
    .urg-high {{ background-color:#ffe1da; color:#b94c35; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }}
    
    /* Hiding Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Penyesuaian Header Teks */
    h1, h2, h3, h4 {{ color: {DARK_TEXT} !important; font-weight: 700 !important; }}
    </style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI BANTU (DATABASE & MODEL)
# ============================================================
def init_db() -> None:
    if not DB_FILE.exists():
        pd.DataFrame(columns=DB_COLUMNS).to_csv(DB_FILE, index=False)
        return
    try:
        df = pd.read_csv(DB_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=DB_COLUMNS)
    except pd.errors.ParserError:
        backup_path = DB_FILE.with_name(DB_FILE.stem + "_backup" + DB_FILE.suffix)
        DB_FILE.rename(backup_path)
        pd.DataFrame(columns=DB_COLUMNS).to_csv(DB_FILE, index=False)
        return

    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing or list(df.columns) != DB_COLUMNS:
        for col in missing:
            df[col] = None
        df[DB_COLUMNS].to_csv(DB_FILE, index=False)

def load_log() -> pd.DataFrame:
    init_db()
    try:
        df = pd.read_csv(DB_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=DB_COLUMNS)
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    return df[DB_COLUMNS]

def append_log(records: list) -> None:
    if not records:
        return
    init_db()
    df_new = pd.DataFrame(records)[DB_COLUMNS]
    header = (not DB_FILE.exists()) or DB_FILE.stat().st_size == 0
    df_new.to_csv(DB_FILE, mode="a", header=header, index=False)

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if YOLO is None or not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception:
        return None

def get_lesion_info(nama: str) -> dict:
    info = LESION_INFO.get(str(nama).lower().strip())
    if info:
        return info
    return {
        "nama_klinis": str(nama).title(),
        "deskripsi": "Informasi klinis belum tersedia untuk kelas ini.",
        "rekomendasi": "Konsultasikan dengan dokter gigi penanggung jawab.",
        "urgensi": "Tidak diketahui",
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    col_brand1, col_brand2 = st.columns([1, 4])
    with col_brand1:
        st.markdown(f"<div style='width:40px; height:40px; background-color:{PRIMARY}; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-size:24px; font-weight:bold;'>🦷</div>", unsafe_allow_html=True)
    with col_brand2:
        st.markdown(f"<h3 style='margin:0; padding:0; color:white !important; font-size:1.2rem;'>{CLINIC_NAME}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; padding:0; font-size:0.8rem; color:#AEC1C3 !important;'>AI Dental Vision System</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    menu = st.radio(
        "Menu Navigasi",
        ["Dashboard", "Riwayat Deteksi", "Analitik", "Referensi Lesi", "Tentang Aplikasi", "Kontak"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; color:#AEC1C3 !important; text-transform:uppercase; font-weight:700; letter-spacing:1px;'>⚙️ Pengaturan Model</p>", unsafe_allow_html=True)
    conf_threshold = st.slider("Ambang Confidence", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("Ambang IoU (NMS)", 0.05, 0.95, 0.45, 0.05)

    st.markdown("<div style='margin-top: 50px; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px;'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:white !important; font-weight:600; font-size:0.9rem; margin:0;'>👨‍⚕️ {USER_NAME}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#AEC1C3 !important; font-size:0.75rem; margin:0;'>{USER_ROLE}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"<div style='text-align:center; margin-top:10px; color:#5E7075 !important;'>Versi {APP_VERSION}</div>", unsafe_allow_html=True)

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard":
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"<p style='color:{PRIMARY_DARK}; font-weight:700; font-size:0.9rem; margin:0; text-transform:uppercase; letter-spacing:1px;'>Klinik AI RSGM</p>", unsafe_allow_html=True)
        st.title("Sistem Skrining Lesi Oral")
        st.markdown(f"<p style='color:#5E7075; font-size:1rem; margin-top:-10px;'>Analisis awal berbasis gambar untuk membantu dokumentasi dan pemantauan klinis.</p>", unsafe_allow_html=True)
    with col_header2:
        st.markdown(f"<div style='text-align:right; margin-top:20px;'><span style='background:#FFF3D6; color:#956A09; padding:6px 16px; border-radius:20px; font-weight:700; font-size:0.85rem;'>📅 {datetime.now().strftime('%d %b %Y')}</span></div>", unsafe_allow_html=True)

    # Banner Peringatan Klinis ala Canva
    st.markdown("""
        <div style="background-color: #FFF9E8; border: 1px solid #E9C46A; border-radius: 12px; padding: 16px; display: flex; align-items: flex-start; gap: 12px; margin-bottom: 24px;">
            <div style="font-size: 20px;">🛡️</div>
            <p style="color: #7C641D; margin: 0; font-size: 0.95rem; line-height: 1.5;"><strong>Peringatan klinis:</strong> Hasil AI hanya merupakan alat bantu skrining awal, bukan diagnosis definitif. Konfirmasi klinis oleh dokter gigi tetap diwajibkan.</p>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model YOLO ('best.pt') tidak ditemukan atau gagal dimuat.")

    df_log = load_log()
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0%"

    # 3 Kartu Metrik Modern
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p class="metric-label">TOTAL PEMERIKSAAN AI</p>
                <span style="font-size:1.2rem;">🔍</span>
            </div>
            <p class="metric-value">{total_deteksi}</p>
            <p style="color:{PRIMARY}; font-size:0.85rem; margin:10px 0 0 0; font-weight:600;">▲ {deteksi_hari_ini} deteksi hari ini</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p class="metric-label">RATA-RATA CONFIDENCE</p>
                <span style="font-size:1.2rem;">🎯</span>
            </div>
            <p class="metric-value">{avg_conf}</p>
            <p style="color:{ACCENT}; font-size:0.85rem; margin:10px 0 0 0; font-weight:600;">Berdasarkan bobot model saat ini</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        status_text = "Database Sinkron" if model else "Model Error"
        status_color = PRIMARY if model else DANGER
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <p class="metric-label">STATUS INTEGRASI</p>
                <span style="font-size:1.2rem;">⚡</span>
            </div>
            <p class="metric-value" style="color:{status_color}; font-size:1.8rem;">{status_text}</p>
            <p style="color:#5E7075; font-size:0.85rem; margin:10px 0 0 0; font-weight:600;">Modul backend aktif</p>
        </div>""", unsafe_allow_html=True)

    # Modul Analisis
    st.markdown("<div style='background:white; padding:24px; border-radius:16px; box-shadow:0 4px 6px rgba(0,0,0,0.02); margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom:5px;'>Analisis Gambar Oral</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5E7075; font-size:0.95rem;'>Unggah satu atau beberapa gambar intraoral (Galeri) atau ambil langsung menggunakan Kamera.</p>", unsafe_allow_html=True)
    
    tab_unggah, tab_kamera = st.tabs(["📁 Unggah dari Galeri/Penyimpanan", "📸 Ambil via Kamera (Mobile/Webcam)"])
    
    images_to_process = []
    file_names = []

    with tab_unggah:
        uploaded_files = st.file_uploader(
            "Seret dan lepas (drag & drop) gambar di sini",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="gallery_uploader"
        )
        if uploaded_files:
            for f in uploaded_files:
                try:
                    img = Image.open(f).convert("RGB")
                    images_to_process.append(img)
                    file_names.append(f.name)
                except Exception:
                    st.warning(f"⚠️ Gagal membuka {f.name}")

    with tab_kamera:
        camera_file = st.camera_input("Ambil gambar lesi langsung")
        if camera_file is not None:
            try:
                img = Image.open(camera_file).convert("RGB")
                images_to_process.append(img)
                file_names.append("Kamera_" + datetime.now().strftime("%H%M%S") + ".jpg")
            except Exception:
                st.warning("⚠️ Gagal memproses gambar kamera.")

    if images_to_process:
        st.markdown("---")
        st.markdown(f"**{len(images_to_process)} gambar siap dianalisis**")
        
        # Tampilkan Pratinjau
        preview_cols = st.columns(min(len(images_to_process), 5))
        for i, img in enumerate(images_to_process[:5]):
            with preview_cols[i]:
                st.image(img, use_container_width=True, caption=file_names[i])
        
        analyze_btn = st.button("🚀 Mulai Analisis YOLO", use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            progress = st.progress(0, text="Memulai analisis...")
            all_new_records = []
            total_lesi = 0

            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress.progress((idx + 1) / len(images_to_process), text=f"Memindai {f_name}...")

                try:
                    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                except Exception as e:
                    st.error(f"⚠️ Inferensi gagal: {e}")
                    continue

                res_plotted = results[0].plot()
                boxes = results[0].boxes
                waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tanggal_sekarang = datetime.now().strftime("%Y-%m-%d")

                detections = []
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    nama_lesi = model.names[class_id]
                    detections.append((nama_lesi, conf_score))
                    all_new_records.append({
                        "ID": str(uuid.uuid4())[:8],
                        "Waktu": waktu_sekarang,
                        "Tanggal": tanggal_sekarang,
                        "Lesi_Terdeteksi": nama_lesi,
                        "Confidence": round(conf_score, 4),
                        "Model_Version": MODEL_PATH.name,
                        "Nama_File": f_name,
                    })
                total_lesi += len(detections)

                with st.expander(f"Hasil Analisis: {f_name} ({len(detections)} temuan)", expanded=True):
                    col_res1, col_res2 = st.columns(2, gap="large")
                    with col_res1:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.markdown("<strong>📸 Citra Masukan</strong>", unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col_res2:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.markdown("<strong>🧬 Identifikasi Bounding Box</strong>", unsafe_allow_html=True)
                        st.image(res_plotted, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if not detections:
                        st.info("Jaringan tampak sehat. Tidak ada anomali yang melebihi ambang batas confidence.")
                    else:
                        st.markdown("##### Rincian Temuan Klinis")
                        for nama_lesi, conf_score in sorted(detections, key=lambda x: -x[1]):
                            info = get_lesion_info(nama_lesi)
                            
                            # Styling Badge Urgensi
                            urg = info['urgensi']
                            badge_class = "urg-high" if "Tinggi" in urg else "urg-med" if urg == "Sedang" else "urg-low"
                            
                            st.markdown(
                                f"""<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:16px; margin-bottom:12px; border-left: 5px solid {PRIMARY};'>
                                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                                        <span style='font-weight:700; font-size:1.1rem; color:{DARK_TEXT};'>{info['nama_klinis']}</span>
                                        <div>
                                            <span style='background:#264653; color:white; padding:4px 10px; border-radius:20px; font-size:0.75rem; font-weight:bold; margin-right:5px;'>{conf_score*100:.1f}%</span>
                                            <span class='{badge_class}'>{info['urgensi']}</span>
                                        </div>
                                    </div>
                                    <p style='font-size:0.9rem; color:#475569; margin-bottom:8px;'>{info['deskripsi']}</p>
                                    <div style='background:white; padding:10px; border-radius:8px; font-size:0.85rem; color:{DARK_TEXT};'>
                                        <strong>Rekomendasi:</strong> {info['rekomendasi']}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

            progress.progress(1.0, text="Selesai.")
            append_log(all_new_records)

            if total_lesi > 0:
                st.success(f"✅ Analisis selesai. {total_lesi} data anomali berhasil direkam ke dalam database.")
            else:
                st.info("Analisis selesai. Tidak ada log yang disimpan.")
    
    st.markdown("</div>", unsafe_allow_html=True) # End Module Container


# ============================================================
# HALAMAN LAINNYA
# ============================================================
elif menu == "Riwayat Deteksi":
    st.title("Riwayat Data Pasien")
    st.markdown("<p style='color:#5E7075;'>Data log deteksi yang tersimpan di dalam sistem.</p>", unsafe_allow_html=True)
    df_log = load_log()

    if df_log.empty:
        st.info("Belum ada data deteksi yang tercatat di database.")
    else:
        st.markdown("<div style='background:white; padding:20px; border-radius:16px; border:1px solid #E2E8F0; margin-bottom:20px;'>", unsafe_allow_html=True)
        st.markdown("**Filter Data**")
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            lesi_options = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Berdasarkan Jenis Lesi", lesi_options, default=lesi_options)
        with col_f2:
            tanggal_series = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            if not tanggal_series.empty:
                min_date, max_date = tanggal_series.min(), tanggal_series.max()
                date_range = st.date_input("Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
        with col_f3:
            min_conf = st.slider("Minimal Confidence", 0.0, 1.0, 0.0, 0.05)

        st.markdown("</div>", unsafe_allow_html=True)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= min_conf)
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            tgl_dt = pd.to_datetime(df_log["Tanggal"], errors="coerce")
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (tgl_dt >= start) & (tgl_dt <= end)

        df_filtered = df_log[mask]
        st.dataframe(df_filtered, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("⬇️ Ekspor Log ke CSV", data=df_filtered.to_csv(index=False).encode("utf-8"), file_name="Riwayat_Deteksi.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_filtered.to_excel(writer, index=False, sheet_name="Data")
                st.download_button("⬇️ Ekspor Log ke Excel", data=buffer.getvalue(), file_name="Riwayat_Deteksi.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            except ImportError:
                st.caption("Pustaka `openpyxl` tidak ditemukan untuk fitur Excel.")

elif menu == "Analitik":
    st.title("Analitik & Performa")
    df_log = load_log()
    if df_log.empty:
        st.info("Belum ada data untuk dianalisis.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<div class='metric-card'><h4>Distribusi Lesi</h4>", unsafe_allow_html=True)
            st.bar_chart(df_log["Lesi_Terdeteksi"].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)
        with col_s2:
            st.markdown("<div class='metric-card'><h4>Tren Harian</h4>", unsafe_allow_html=True)
            st.line_chart(df_log.groupby("Tanggal").size())
            st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Referensi Lesi":
    st.title("Referensi Klinis Lesi Oral")
    st.markdown("<p style='color:#5E7075;'>Ringkasan edukasi untuk mendampingi hasil skrining model AI.</p>", unsafe_allow_html=True)
    
    for key, info in LESION_INFO.items():
        urg = info['urgensi']
        badge_class = "urg-high" if "Tinggi" in urg else "urg-med" if urg == "Sedang" else "urg-low"
        
        st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px; border:1px solid #E2E8F0; margin-bottom:15px; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #E2E8F0; padding-bottom:10px; margin-bottom:10px;">
                    <h4 style="margin:0;">{info['nama_klinis']}</h4>
                    <span class='{badge_class}'>{info['urgensi']}</span>
                </div>
                <p style="color:#475569; font-size:0.95rem;">{info['deskripsi']}</p>
                <div style="background:#F4F7F7; padding:12px; border-radius:8px; font-size:0.9rem;">
                    <strong>Rekomendasi Klinis:</strong> {info['rekomendasi']}
                </div>
            </div>
        """, unsafe_allow_html=True)

elif menu == "Tentang Aplikasi":
    st.title("Informasi Sistem")
    st.info("Sistem ini diintegrasikan menggunakan backend **Python Streamlit** dan **YOLOv11** dengan antarmuka CSS kustom yang diadaptasi dari desain modern.")
    st.write("**Catatan:** Seluruh data pasien disimulasikan sebagai log anonim (ID *hash*) dan sistem didesain sebagai penunjang keputusan tahap awal.")

elif menu == "Kontak":
    st.title("Hubungi Administrator")
    st.markdown("""
    <div style='background:white; padding:30px; border-radius:16px; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
        <div style='font-size:3rem; margin-bottom:10px;'>🏥</div>
        <h3>Dukungan Teknis & Klinis</h3>
        <p style='color:#5E7075;'>Silakan hubungi administrator IT Fakultas/Klinik untuk pembaruan bobot model (.pt) atau reset database CSV.</p>
    </div>
    """, unsafe_allow_html=True)
