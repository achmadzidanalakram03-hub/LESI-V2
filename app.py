"""
Klinik AI RSGM — Sistem Skrining Lesi Oral berbasis YOLO
=========================================================
Antarmuka Minimalis, Clean, dan Modern SaaS
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
APP_VERSION = "2.1 (Clean UI)"
CLINIC_NAME = "RSGM Unjani"
USER_NAME = "drg. Adinara Savero, S.KG"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi.csv")
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi",
    "Confidence", "Model_Version", "Nama_File",
]

# PALET WARNA (Minimalist Medical SaaS)
PRIMARY = "#0f766e"       # Deep Teal
PRIMARY_LIGHT = "#ccfbf1"
ACCENT = "#fbbf24"        # Amber
DANGER = "#ef4444"        # Red
DARK_TEXT = "#0f172a"     # Slate 900
GRAY_TEXT = "#64748b"     # Slate 500
BG_MIST = "#f8fafc"       # Slate 50
BORDER_COLOR = "#e2e8f0"

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

# --- KONFIGURASI HALAMAN & INJEKSI CSS ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: {BG_MIST}; }}
    
    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6 {{ color: {DARK_TEXT} !important; font-weight: 600 !important; letter-spacing: -0.02em; }}
    p {{ color: {GRAY_TEXT}; }}
    
    /* Clean Metric Cards */
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }}
    .metric-label {{
        font-size: 0.875rem; 
        color: {GRAY_TEXT};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }}
    .metric-value {{ 
        font-size: 2.25rem; 
        font-weight: 600; 
        color: {DARK_TEXT}; 
        line-height: 1.1; 
        margin: 0; 
    }}
    .metric-sub {{ font-size: 0.875rem; font-weight: 500; margin-top: 8px; }}
    
    /* Result Container */
    .result-container {{
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
        margin-top: 24px;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {PRIMARY} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton>button:hover {{ background-color: #115e59 !important; box-shadow: 0 4px 6px -1px rgba(15,23,42,0.1) !important; }}
    
    /* Badges */
    .badge {{ padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: inline-block; }}
    .badge-low {{ background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
    .badge-med {{ background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
    .badge-high {{ background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
    .badge-score {{ background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; margin-right: 8px; }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ background-color: white !important; border-right: 1px solid {BORDER_COLOR}; }}
    [data-testid="stSidebar"] * {{ color: {DARK_TEXT}; }}
    
    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI BANTU
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
    if not records: return
    init_db()
    df_new = pd.DataFrame(records)[DB_COLUMNS]
    header = (not DB_FILE.exists()) or DB_FILE.stat().st_size == 0
    df_new.to_csv(DB_FILE, mode="a", header=header, index=False)

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if YOLO is None or not path.exists(): return None
    try: return YOLO(str(path))
    except Exception: return None

def get_lesion_info(nama: str) -> dict:
    info = LESION_INFO.get(str(nama).lower().strip())
    if info: return info
    return {
        "nama_klinis": str(nama).title(),
        "deskripsi": "Informasi klinis belum tersedia untuk anomali ini.",
        "rekomendasi": "Evaluasi klinis mendalam oleh dokter penanggung jawab.",
        "urgensi": "Tidak Diketahui",
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="margin-bottom: 32px;">
            <h2 style="margin:0; font-size: 1.25rem; color: {PRIMARY} !important;">{CLINIC_NAME}</h2>
            <p style="margin:0; font-size: 0.875rem;">AI Vision Screening</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", ["Dashboard", "Riwayat Deteksi", "Analitik", "Referensi Lesi", "Sistem"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("**Pengaturan Parameter**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU (NMS) Threshold", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 40px; padding: 16px; background: {BG_MIST}; border-radius: 8px; border: 1px solid {BORDER_COLOR};">
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT}; text-transform: uppercase;">Operator Aktif</p>
            <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 0.875rem;">{USER_NAME}</p>
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT};">{USER_ROLE}</p>
        </div>
    """, unsafe_allow_html=True)

model = load_model(MODEL_PATH)

# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard":
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"<h1 style='margin-bottom: 4px;'>Analisis Lesi Intraoral</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; margin-top: 0;'>Deteksi dini dan dokumentasi klinis berbasis Computer Vision.</p>", unsafe_allow_html=True)
    with col_header2:
        st.markdown(f"<div style='text-align: right; margin-top: 16px; font-weight: 500; color: {GRAY_TEXT};'>{datetime.now().strftime('%d %b %Y')}</div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: {PRIMARY_LIGHT}; border: 1px solid #5eead4; border-left: 4px solid {PRIMARY}; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
            <p style="margin: 0; color: #115e59; font-size: 0.9rem;">
                <strong>Disclaimer Medis:</strong> Hasil analisis ini bersifat penunjang. Keputusan diagnosis akhir tetap berada pada kewenangan profesional medis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    df_log = load_log()
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0.0%"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Pemeriksaan</div>
            <p class="metric-value">{total_deteksi}</p>
            <p class="metric-sub" style="color: {PRIMARY};">+{deteksi_hari_ini} tercatat hari ini</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-rata Akurasi (Conf)</div>
            <p class="metric-value">{avg_conf}</p>
            <p class="metric-sub" style="color: {GRAY_TEXT};">Berdasarkan inferensi model</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        status_text = "Sistem Aktif" if model else "Model Offline"
        status_color = PRIMARY if model else DANGER
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status Integrasi AI</div>
            <p class="metric-value" style="color: {status_color};">{status_text}</p>
            <p class="metric-sub" style="color: {GRAY_TEXT};">Backend tersinkronisasi</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("### Modul Akuisisi Gambar")
    
    tab_unggah, tab_kamera = st.tabs(["Unggah Berkas", "Kamera Perangkat"])
    images_to_process, file_names = [], []

    with tab_unggah:
        uploaded_files = st.file_uploader("Pilih file gambar intraoral", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    images_to_process.append(Image.open(f).convert("RGB"))
                    file_names.append(f.name)
                except Exception: pass

    with tab_kamera:
        camera_file = st.camera_input("Ambil gambar secara langsung")
        if camera_file is not None:
            try:
                images_to_process.append(Image.open(camera_file).convert("RGB"))
                file_names.append("Cam_" + datetime.now().strftime("%H%M%S") + ".jpg")
            except Exception: pass

    if images_to_process:
        analyze_btn = st.button("Mulai Proses Inferensi YOLO", use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            progress = st.progress(0, text="Menginisialisasi analisis...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress.progress((idx + 1) / len(images_to_process), text=f"Memproses {f_name}...")
                try:
                    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                except Exception as e:
                    st.error(f"Gagal memproses {f_name}: {e}")
                    continue

                res_plotted = results[0].plot()
                boxes = results[0].boxes
                detections = []
                
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    nama_lesi = model.names[class_id]
                    detections.append((nama_lesi, conf_score))
                    all_new_records.append({
                        "ID": str(uuid.uuid4())[:8],
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Tanggal": datetime.now().strftime("%Y-%m-%d"),
                        "Lesi_Terdeteksi": nama_lesi,
                        "Confidence": round(conf_score, 4),
                        "Model_Version": MODEL_PATH.name,
                        "Nama_File": f_name,
                    })

                # Tampilan Sejajar Hasil (Clean Layout)
                st.markdown(f"<div class='result-container'>", unsafe_allow_html=True)
                st.markdown(f"#### Hasil Analisis: {f_name}")
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("<p style='font-size: 0.875rem; font-weight: 500; margin-bottom: 8px;'>Citra Asli</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_img2:
                    st.markdown("<p style='font-size: 0.875rem; font-weight: 500; margin-bottom: 8px;'>Pemetaan Bounding Box</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                if not detections:
                    st.info("Tidak terdeteksi adanya anomali klinis berdasarkan parameter ambang batas.")
                else:
                    st.markdown("<hr style='margin: 24px 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)
                    st.markdown("##### Rincian Identifikasi")
                    for nama_lesi, conf_score in sorted(detections, key=lambda x: -x[1]):
                        info = get_lesion_info(nama_lesi)
                        urg = info['urgensi']
                        badge_class = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
                        
                        st.markdown(f"""
                            <div style="background: {BG_MIST}; border-radius: 8px; padding: 16px; margin-bottom: 12px; border: 1px solid {BORDER_COLOR}; border-left: 4px solid {PRIMARY};">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="font-weight: 600; font-size: 1rem; color: {DARK_TEXT};">{info['nama_klinis']}</span>
                                    <div>
                                        <span class="badge badge-score">Conf: {conf_score*100:.1f}%</span>
                                        <span class="badge {badge_class}">Urgensi: {info['urgensi']}</span>
                                    </div>
                                </div>
                                <p style="font-size: 0.875rem; color: {GRAY_TEXT}; margin-bottom: 12px;">{info['deskripsi']}</p>
                                <div style="background: white; border-radius: 6px; padding: 12px; border: 1px solid {BORDER_COLOR}; font-size: 0.875rem;">
                                    <strong>Tindakan Medis:</strong> {info['rekomendasi']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            progress.progress(1.0, text="Selesai.")
            append_log(all_new_records)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HALAMAN LAINNYA
# ============================================================
elif menu == "Riwayat Deteksi":
    st.markdown("## Database Pemeriksaan")
    st.markdown("Arsip historis deteksi klinis yang terekam dalam sistem lokal.")
    df_log = load_log()

    if df_log.empty:
        st.info("Basis data log saat ini kosong.")
    else:
        st.markdown("<div style='background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px;'>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            opts = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Filter Jenis Lesi", opts, default=opts)
        with col_f2:
            tgl_series = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            date_range = st.date_input("Rentang Waktu", value=(tgl_series.min(), tgl_series.max())) if not tgl_series.empty else None
        with col_f3:
            min_conf = st.slider("Batas Minimum Confidence", 0.0, 1.0, 0.0)
        st.markdown("</div>", unsafe_allow_html=True)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= min_conf)
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)

        df_filtered = df_log[mask]
        st.dataframe(df_filtered, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("Ekspor Format CSV", data=df_filtered.to_csv(index=False).encode("utf-8"), file_name="Riwayat_Klinis.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("Ekspor Format Excel", data=buffer.getvalue(), file_name="Riwayat_Klinis.xlsx", use_container_width=True)
            except ImportError: pass

elif menu == "Analitik":
    st.markdown("## Tinjauan Analitik")
    df_log = load_log()
    if df_log.empty: st.info("Tidak ada data analitik tersedia.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<div class='metric-card'><div class='metric-label'>Distribusi Prevalensi Lesi</div>", unsafe_allow_html=True)
            st.bar_chart(df_log["Lesi_Terdeteksi"].value_counts())
            st.markdown("</div>", unsafe_allow_html=True)
        with col_s2:
            st.markdown("<div class='metric-card'><div class='metric-label'>Tren Skrining Harian</div>", unsafe_allow_html=True)
            st.line_chart(df_log.groupby("Tanggal").size())
            st.markdown("</div>", unsafe_allow_html=True)

elif menu == "Referensi Lesi":
    st.markdown("## Ensiklopedia Lesi Oral")
    for key, info in LESION_INFO.items():
        badge_class = "badge-high" if "Tinggi" in info['urgensi'] else "badge-med" if info['urgensi'] == "Sedang" else "badge-low"
        st.markdown(f"""
            <div style="background: white; border-radius: 12px; padding: 24px; border: 1px solid {BORDER_COLOR}; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 12px; margin-bottom: 12px;">
                    <h4 style="margin: 0; color: {PRIMARY} !important;">{info['nama_klinis']}</h4>
                    <span class="badge {badge_class}">{info['urgensi']}</span>
                </div>
                <p style="color: {GRAY_TEXT}; margin-bottom: 16px;">{info['deskripsi']}</p>
                <div style="background: {BG_MIST}; padding: 12px 16px; border-radius: 8px; font-size: 0.875rem;">
                    <strong>Panduan Klinis:</strong> {info['rekomendasi']}
                </div>
            </div>
        """, unsafe_allow_html=True)

elif menu == "Sistem":
    st.markdown("## Informasi Infrastruktur")
    st.markdown(f"""
    <div style='background: white; padding: 32px; border-radius: 12px; border: 1px solid {BORDER_COLOR};'>
        <h4 style="margin-top:0;">Spesifikasi Deployment</h4>
        <ul style="color: {GRAY_TEXT};">
            <li><strong>Framework:</strong> Streamlit / Python 3</li>
            <li><strong>Architecture:</strong> YOLO by Ultralytics</li>
            <li><strong>Environment:</strong> Frontend terintegrasi backend lokal</li>
        </ul>
        <hr style="border-color: {BORDER_COLOR}; margin: 24px 0;">
        <h4>Kontak Dukungan</h4>
        <p style="color: {GRAY_TEXT};">Pembaruan parameter bobot atau manajemen instans dapat dilakukan melalui administrator server utama institusi terkait.</p>
    </div>
    """, unsafe_allow_html=True)
