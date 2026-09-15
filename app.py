"""
Klinik AI RSGM — Sistem Skrining Lesi Oral berbasis YOLO
=========================================================
Terintegrasi dengan Anamnesis OLD CARTS & Sidebar Frost UI
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
# KONFIGURASI GLOBAL & DATABASE
# ============================================================
APP_VERSION = "3.0 (Frost UI & EMR)"
CLINIC_NAME = "RSGM Unjani"
USER_NAME = "drg. Adinara Savero, S.KG"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi.csv")

# Penambahan kolom OLD CARTS untuk Electronic Medical Record (EMR)
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi", "Confidence", "Nama_File",
    "Onset", "Location", "Duration", "Character", "Aggravating", "Relieving", "Timing", "Severity"
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
    "cheek biting": {"nama_klinis": "Morsicatio Buccarum", "urgensi": "Rendah"},
    "coated tongue": {"nama_klinis": "Coated Tongue", "urgensi": "Rendah"},
    "karies": {"nama_klinis": "Karies Gigi", "urgensi": "Sedang-Tinggi"},
    "linea alba": {"nama_klinis": "Linea Alba", "urgensi": "Rendah"},
    "lingual varicosites": {"nama_klinis": "Lingual Varicosities", "urgensi": "Rendah"},
    "stain calculus": {"nama_klinis": "Stain & Kalkulus", "urgensi": "Sedang"},
    "torus": {"nama_klinis": "Torus", "urgensi": "Rendah"},
    "ulkus traumatikus": {"nama_klinis": "Ulkus Traumatikus", "urgensi": "Sedang"},
}

# Inisialisasi Session State untuk Anamnesis
if 'anamnesis_data' not in st.session_state:
    st.session_state.anamnesis_data = None


# --- KONFIGURASI HALAMAN & INJEKSI CSS (FROST UI) ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    
    /* Latar belakang utama dengan gradien subtil agar efek Frost UI Sidebar lebih menonjol */
    .stApp {{ 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }}
    
    /* FROST UI / GLASSMORPHISM UNTUK SIDEBAR */
    [data-testid="stSidebar"] {{ 
        background: rgba(255, 255, 255, 0.65) !important; 
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important; /* Dukungan untuk iOS/Mac Safari */
        border-right: 1px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.05);
    }}
    [data-testid="stSidebar"] * {{ color: {DARK_TEXT}; }}
    
    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6 {{ color: {DARK_TEXT} !important; font-weight: 600 !important; letter-spacing: -0.02em; }}
    p {{ color: {GRAY_TEXT}; }}
    
    /* Containers */
    .clean-card {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
    }}
    
    /* Metric Cards */
    .metric-label {{ font-size: 0.875rem; color: {GRAY_TEXT}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
    .metric-value {{ font-size: 2.25rem; font-weight: 600; color: {DARK_TEXT}; line-height: 1.1; margin: 0; }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {PRIMARY} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2) !important;
    }}
    .stButton>button:hover {{ background-color: #115e59 !important; box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3) !important; transform: translateY(-1px); }}
    
    /* Badges */
    .badge {{ padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: inline-block; }}
    .badge-low {{ background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
    .badge-med {{ background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
    .badge-high {{ background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI BANTU & LOGIKA KECERDASAN KLINIS
# ============================================================
def init_db() -> None:
    if not DB_FILE.exists():
        pd.DataFrame(columns=DB_COLUMNS).to_csv(DB_FILE, index=False)
        return
    try:
        df = pd.read_csv(DB_FILE)
    except Exception:
        df = pd.DataFrame(columns=DB_COLUMNS)

    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing or list(df.columns) != DB_COLUMNS:
        for col in missing:
            df[col] = None
        df[DB_COLUMNS].to_csv(DB_FILE, index=False)

def load_log() -> pd.DataFrame:
    init_db()
    try:
        df = pd.read_csv(DB_FILE)
    except Exception:
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

def synthesize_clinical_diagnosis(detections, anamnesis):
    """Logika kecerdasan untuk menggabungkan deteksi visual dan anamnesis."""
    if not detections:
        return "Tidak ada anomali visual yang terdeteksi. Observasi klinis lanjutan disarankan berdasarkan keluhan pasien."
    
    suspek_list = []
    
    for nama_lesi, conf in detections:
        lesi_key = nama_lesi.lower()
        
        if anamnesis:
            sev = anamnesis.get("Severity", 0)
            char = str(anamnesis.get("Character", "")).lower()
            
            # Logika Karies
            if lesi_key == "karies":
                if sev >= 7 or "denyut" in char:
                    suspek_list.append("Suspek Pulpitis Irreversibel / Apikalis Akut (Berdasarkan deteksi karies profil tinggi dipadu skala nyeri berat/berdenyut).")
                elif sev > 0:
                    suspek_list.append("Suspek Pulpitis Reversibel (Karies disertai keluhan nyeri ringan-sedang).")
                else:
                    suspek_list.append("Karies Asimtomatik (Terkonfirmasi secara visual tanpa keluhan nyeri terkait).")
            
            # Logika Ulkus / Cheek Biting
            elif lesi_key in ["ulkus traumatikus", "cheek biting"]:
                if sev > 4:
                    suspek_list.append(f"Stomatitis / Ulserasi Traumatik Simtomatik (Tingkat nyeri {sev}/10). Perlu eliminasi faktor traumatik.")
                else:
                    suspek_list.append(f"{LESION_INFO[lesi_key]['nama_klinis']} ringan. Cenderung dapat sembuh spontan.")
            
            # Logika Stain / Calculus
            elif lesi_key == "stain calculus":
                if "darah" in char or "bengkak" in char:
                    suspek_list.append("Suspek Gingivitis / Periodontitis (Kalkulus terdeteksi disertai keluhan spesifik periodontal).")
                else:
                    suspek_list.append("Deposit Kalkulus / Stain tanpa komplikasi akut berlebih.")
            
            else:
                suspek_list.append(f"Korelasi klinis ditemukan untuk {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            suspek_list.append(f"Deteksi visual murni: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
            
    return " | ".join(suspek_list)


# ============================================================
# SIDEBAR (FROST UI)
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="margin-bottom: 32px;">
            <h2 style="margin:0; font-size: 1.25rem; color: {PRIMARY} !important;">{CLINIC_NAME}</h2>
            <p style="margin:0; font-size: 0.875rem;">Sistem Rekam Medis & AI</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", ["Dashboard Skrining", "Riwayat Rekam Medis", "Ensiklopedia Lesi"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("**Pengaturan AI Visual**")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 40px; padding: 16px; background: rgba(255,255,255,0.5); border-radius: 8px; border: 1px solid rgba(0,0,0,0.05);">
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT}; text-transform: uppercase;">Operator Aktif</p>
            <p style="margin: 4px 0 0 0; font-weight: 600; font-size: 0.875rem;">{USER_NAME}</p>
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT};">{USER_ROLE}</p>
        </div>
    """, unsafe_allow_html=True)

model = load_model(MODEL_PATH)

# ============================================================
# HALAMAN: DASHBOARD & OLD CARTS
# ============================================================
if menu == "Dashboard Skrining":
    st.markdown(f"<h1 style='margin-bottom: 4px;'>Dashboard Skrining Terpadu</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; margin-top: 0; margin-bottom: 24px;'>Integrasi Anamnesis Subjektif dan Deteksi Computer Vision.</p>", unsafe_allow_html=True)

    # --- TAHAP 1: FORMULIR ANAMNESIS (OLD CARTS) ---
    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Tahap 1: Anamnesis (Metode OLD CARTS)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.875rem;'>Pengisian anamnesis akan membantu sistem memberikan saran diagnosis suspek yang lebih akurat.</p>", unsafe_allow_html=True)
    
    with st.form("form_old_carts"):
        col_oc1, col_oc2 = st.columns(2)
        with col_oc1:
            onset = st.text_input("Onset (O)", placeholder="Kapan gejala pertama kali muncul?")
            location = st.text_input("Location (L)", placeholder="Di bagian mana keluhan terasa?")
            duration = st.text_input("Duration (D)", placeholder="Berapa lama durasi sakitnya?")
            character = st.selectbox("Character (C)", ["", "Berdenyut", "Tajam menusuk", "Tumpul / Ngilu", "Panas / Terbakar", "Berdarah", "Lainnya"])
        with col_oc2:
            aggravating = st.text_input("Aggravating (A)", placeholder="Faktor yang memperparah?")
            relieving = st.text_input("Relieving (R)", placeholder="Faktor yang meredakan?")
            timing = st.selectbox("Timing (T)", ["", "Pagi hari", "Malam hari", "Setelah makan", "Terus-menerus", "Hilang timbul"])
            severity = st.slider("Severity (S) - Skala Nyeri", 0, 10, 0)
        
        submit_anamnesis = st.form_submit_button("Simpan Data Anamnesis", use_container_width=True)
        
        if submit_anamnesis:
            st.session_state.anamnesis_data = {
                "Onset": onset, "Location": location, "Duration": duration,
                "Character": character, "Aggravating": aggravating,
                "Relieving": relieving, "Timing": timing, "Severity": severity
            }
            st.success("Data anamnesis berhasil direkam ke dalam memori sesi sementara.")
    
    if st.session_state.anamnesis_data:
        st.markdown(f"""
        <div style="background: {PRIMARY_LIGHT}; padding: 12px 16px; border-radius: 8px; border-left: 4px solid {PRIMARY}; font-size: 0.875rem; color: #115e59; margin-bottom: 20px;">
            <strong>Status Anamnesis:</strong> Data terisi (Skala Nyeri: {st.session_state.anamnesis_data['Severity']}/10). Melanjutkan ke tahap visual.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- TAHAP 2: DETEKSI VISUAL ---
    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Tahap 2: Akuisisi Citra Klinis</h3>", unsafe_allow_html=True)
    
    tab_unggah, tab_kamera = st.tabs(["Unggah Berkas Gambar", "Kamera Perangkat"])
    images_to_process, file_names = [], []

    with tab_unggah:
        uploaded_files = st.file_uploader("Pilih file gambar intraoral (JPG/PNG)", accept_multiple_files=True)
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
                file_names.append("Kamera_" + datetime.now().strftime("%H%M%S") + ".jpg")
            except Exception: pass

    if images_to_process:
        analyze_btn = st.button("Jalankan Inferensi AI & Sinkronisasi Diagnosis", use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            progress = st.progress(0, text="Menganalisis gambar...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress.progress((idx + 1) / len(images_to_process), text=f"Memproses {f_name}...")
                results = model(image, conf=conf_threshold, verbose=False)
                
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                detections = []
                
                # Ekstrak anamnesis untuk database
                anamnesis = st.session_state.anamnesis_data or {}
                
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    nama_lesi = model.names[class_id]
                    detections.append((nama_lesi, conf_score))
                    
                    record = {
                        "ID": str(uuid.uuid4())[:8],
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Tanggal": datetime.now().strftime("%Y-%m-%d"),
                        "Lesi_Terdeteksi": nama_lesi,
                        "Confidence": round(conf_score, 4),
                        "Nama_File": f_name,
                        "Onset": anamnesis.get("Onset", ""),
                        "Location": anamnesis.get("Location", ""),
                        "Duration": anamnesis.get("Duration", ""),
                        "Character": anamnesis.get("Character", ""),
                        "Aggravating": anamnesis.get("Aggravating", ""),
                        "Relieving": anamnesis.get("Relieving", ""),
                        "Timing": anamnesis.get("Timing", ""),
                        "Severity": anamnesis.get("Severity", None)
                    }
                    all_new_records.append(record)

                # Sintesis Klinis Cerdas
                sintesis_akhir = synthesize_clinical_diagnosis(detections, anamnesis)

                # Render Hasil (Sejajar)
                st.markdown(f"<div style='margin-top: 32px;'><h4>Laporan Klinis: {f_name}</h4></div>", unsafe_allow_html=True)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("<p style='font-size: 0.875rem; font-weight: 500; margin-bottom: 8px;'>Citra Klinis Asli</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_img2:
                    st.markdown("<p style='font-size: 0.875rem; font-weight: 500; margin-bottom: 8px;'>Deteksi Visual (YOLO)</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                # Kotak Sintesis Diagnosis
                st.markdown(f"""
                    <div style="background: white; border: 1px solid {BORDER_COLOR}; border-left: 4px solid {PRIMARY}; border-radius: 8px; padding: 20px; margin-top: 16px;">
                        <p style="font-size: 0.75rem; text-transform: uppercase; color: {GRAY_TEXT}; font-weight: 600; margin: 0 0 8px 0; letter-spacing: 0.05em;">Sintesis Suspek Diagnosis</p>
                        <p style="font-size: 1rem; color: {DARK_TEXT}; font-weight: 500; margin: 0;">{sintesis_akhir}</p>
                    </div>
                """, unsafe_allow_html=True)

            progress.progress(1.0, text="Selesai merekam data medis.")
            append_log(all_new_records)
            
            # Reset anamnesis setelah disimpan untuk pasien berikutnya
            st.session_state.anamnesis_data = None
            
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: RIWAYAT REKAM MEDIS
# ============================================================
elif menu == "Riwayat Rekam Medis":
    st.markdown("## Rekam Medis Elektronik (EMR)")
    st.markdown("Arsip integrasi data anamnesis OLD CARTS dan deteksi visual AI.")
    
    df_log = load_log()

    if df_log.empty:
        st.info("Database EMR kosong.")
    else:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        tgl_series = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
        date_range = st.date_input("Filter Waktu", value=(tgl_series.min(), tgl_series.max())) if not tgl_series.empty else None
        st.markdown("</div>", unsafe_allow_html=True)

        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask = (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)
            df_filtered = df_log[mask]
        else:
            df_filtered = df_log

        st.dataframe(df_filtered, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("Ekspor Format CSV", data=df_filtered.to_csv(index=False).encode("utf-8"), file_name="EMR_RSGM.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("Ekspor Format Excel", data=buffer.getvalue(), file_name="EMR_RSGM.xlsx", use_container_width=True)
            except ImportError: pass


# ============================================================
# HALAMAN: REFERENSI
# ============================================================
elif menu == "Ensiklopedia Lesi":
    st.markdown("## Standar Referensi Klinis")
    for key, info in LESION_INFO.items():
        badge_class = "badge-high" if "Tinggi" in info['urgensi'] else "badge-med" if info['urgensi'] == "Sedang" else "badge-low"
        st.markdown(f"""
            <div class='clean-card' style="padding: 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 12px; margin-bottom: 12px;">
                    <h4 style="margin: 0; color: {PRIMARY} !important;">{info['nama_klinis']}</h4>
                    <span class="badge {badge_class}">Urgensi: {info['urgensi']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
