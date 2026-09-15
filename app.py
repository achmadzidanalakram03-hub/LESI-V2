"""
Klinik AI RSGM — Sistem Skrining Lesi Oral & Anamnesis Terpadu
=============================================================
Desain UI/UX: Premium SaaS (Split-Screen Login, Dark Modern Sidebar, Fairy Palette)
"""

import io
import uuid
import base64
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
APP_VERSION = "4.3 (Complete EMR & Prevalence Analytics)"
CLINIC_NAME = "RSGM Unjani"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

LOGO_PATH = "Universitas Jenderal Achmad Yani (UNJANI) Logo - Colored - zonalogo.com.jpg" 

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi_emr.csv")

# Seluruh kolom data rekam medis lengkap
DB_COLUMNS = [
    "ID", "Tanggal", "Waktu", "Nama_File", "Lesi_Terdeteksi", "Confidence",
    "O_Onset", "L_Location", "D_Duration", "C_Character", 
    "A_Aggravating", "R_Relieving", "T_Timing", "S_Severity", "Suspek_Diagnosis"
]

# PALET WARNA (Fairy Color Palette & Modern Medical)
C_PURPLE = "#A163F7"
C_BLUE   = "#6F88FC"
C_CYAN   = "#45E3FF"
C_CREAM  = "#F9FAFB" 

BG_MAIN       = "#F4F7F9"
BG_SIDEBAR    = "#1A1D2D" 
TEXT_DARK     = "#1E293B"
TEXT_MUTED    = "#64748B"
BORDER_COLOR  = "#E2E8F0"

LESION_INFO = {
    "cheek biting": {"nama_klinis": "Morsicatio Buccarum", "deskripsi": "Lesi traumatik akibat gigitan berulang pada mukosa pipi.", "rekomendasi": "Edukasi hilangkan habit; evaluasi 2 minggu.", "urgensi": "Rendah"},
    "coated tongue": {"nama_klinis": "Coated Tongue", "deskripsi": "Penumpukan debris keratin pada dorsum lidah.", "rekomendasi": "Instruksi pembersihan dengan tongue scraper.", "urgensi": "Rendah"},
    "karies": {"nama_klinis": "Karies Gigi", "deskripsi": "Demineralisasi jaringan keras gigi akibat bakteri.", "rekomendasi": "Rujuk untuk preparasi dan restorasi/perawatan saluran akar.", "urgensi": "Sedang-Tinggi"},
    "linea alba": {"nama_klinis": "Linea Alba Buccalis", "deskripsi": "Garis hiperkeratosis putih sejajar bidang oklusal.", "rekomendasi": "Lesi jinak, tidak perlu intervensi khusus.", "urgensi": "Rendah"},
    "lingual varicosites": {"nama_klinis": "Lingual Varicosities", "deskripsi": "Pelebaran pembuluh darah vena di ventral lidah.", "rekomendasi": "Observasi klinis, fisiologis pada usia lanjut.", "urgensi": "Rendah"},
    "stain calculus": {"nama_klinis": "Stain & Kalkulus", "deskripsi": "Deposit mineral plak terkalsifikasi.", "rekomendasi": "Tindakan scaling dan root planing (SRP).", "urgensi": "Sedang"},
    "torus": {"nama_klinis": "Torus", "deskripsi": "Eksostosis tulang jinak asimtomatik.", "rekomendasi": "Observasi kecuali mengganggu pembuatan protesa.", "urgensi": "Rendah"},
    "ulkus traumatikus": {"nama_klinis": "Ulkus Traumatikus", "deskripsi": "Hilangnya lapisan epitel akibat trauma.", "rekomendasi": "Kendalikan faktor etiologi, berikan obat topikal.", "urgensi": "Sedang"},
    "olp": {"nama_klinis": "Oral Lichen Planus", "deskripsi": "Kondisi peradangan kronis pada mukosa mulut.", "rekomendasi": "Evaluasi klinis lanjutan, pertimbangkan biopsi jika atipikal.", "urgensi": "Sedang"},
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'anamnesis_data' not in st.session_state:
    st.session_state.anamnesis_data = None

st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# FUNGSI BANTU & DATABASE EMR
# ============================================================
def init_db() -> None:
    """Inisialisasi database CSV dan isi data dummy jika kosong agar grafik & EMR langsung siap."""
    if not DB_FILE.exists() or DB_FILE.stat().st_size == 0:
        dummy_data = [
            {
                "ID": "a1b2c3d4", "Tanggal": datetime.now().strftime("%Y-%m-%d"), "Waktu": "09:30:15", "Nama_File": "klinis_gigi_01.jpg",
                "Lesi_Terdeteksi": "karies", "Confidence": 0.8921, "O_Onset": "3 hari lalu", "L_Location": "Regio gigi molar 36", 
                "D_Duration": "Hilang timbul", "C_Character": "Berdenyut tajam", "A_Aggravating": "Air dingin dan manis", 
                "R_Relieving": "Minum obat analgetik", "T_Timing": "Malam hari", "S_Severity": 7, 
                "Suspek_Diagnosis": "Suspek Pulpitis Irreversibel (Karies profunda + nyeri spontan/berdenyut parah)."
            },
            {
                "ID": "e5f6g7h8", "Tanggal": datetime.now().strftime("%Y-%m-%d"), "Waktu": "10:15:40", "Nama_File": "mukosa_bukal_02.jpg",
                "Lesi_Terdeteksi": "cheek biting", "Confidence": 0.8432, "O_Onset": "1 minggu lalu", "L_Location": "Mukosa bukal kanan", 
                "D_Duration": "Terus-menerus", "C_Character": "Ngilu/Tumpul", "A_Aggravating": "Tergesek makanan", 
                "R_Relieving": "Istirahat", "T_Timing": "Sepanjang waktu", "S_Severity": 3, 
                "Suspek_Diagnosis": "Lesi traumatik tahap penyembuhan / ringan."
            },
            {
                "ID": "i9j0k1l2", "Tanggal": datetime.now().strftime("%Y-%m-%d"), "Waktu": "11:00:22", "Nama_File": "gingiva_03.jpg",
                "Lesi_Terdeteksi": "stain calculus", "Confidence": 0.9150, "O_Onset": "1 bulan lalu", "L_Location": "Regio anterior rahang bawah", 
                "D_Duration": "Kronis", "C_Character": "Berdarah", "A_Aggravating": "Saat menyikat gigi", 
                "R_Relieving": "Berkumur air garam", "T_Timing": "Pagi hari", "S_Severity": 2, 
                "Suspek_Diagnosis": "Suspek Periodontitis/Gingivitis (Kalkulus + keluhan perdarahan gingiva)."
            }
        ]
        pd.DataFrame(dummy_data).to_csv(DB_FILE, index=False)
        return

    try: 
        df = pd.read_csv(DB_FILE)
    except Exception: 
        df = pd.DataFrame(columns=DB_COLUMNS)
        
    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing:
        for col in missing: df[col] = None
        df[DB_COLUMNS].to_csv(DB_FILE, index=False)

def load_log() -> pd.DataFrame:
    init_db()
    try: df = pd.read_csv(DB_FILE)
    except Exception: df = pd.DataFrame(columns=DB_COLUMNS)
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
        "deskripsi": "Informasi klinis belum tersedia di database untuk anomali ini.",
        "rekomendasi": "Lakukan evaluasi klinis mendalam oleh dokter gigi.",
        "urgensi": "Tidak Diketahui",
    }

def synthesize_clinical_diagnosis(detections, anamnesis):
    if not detections:
        return "Tidak teridentifikasi kelainan struktural secara visual. Pendekatan simptomatik disarankan."
    hasil_sintesis = []
    for nama_lesi, _ in detections:
        lesi_key = nama_lesi.lower()
        if anamnesis:
            sev = anamnesis.get("S_Severity", 0)
            char = str(anamnesis.get("C_Character", "")).lower()
            timing = str(anamnesis.get("T_Timing", "")).lower()
            
            if lesi_key == "karies":
                if sev >= 7 or "denyut" in char or "malam" in timing:
                    hasil_sintesis.append("Suspek Pulpitis Irreversibel (Karies profunda + nyeri spontan/berdenyut parah).")
                elif sev >= 4 or "ngilu" in char:
                    hasil_sintesis.append("Suspek Pulpitis Reversibel (Karies terdeteksi + ngilu terpicu stimulus).")
                else:
                    hasil_sintesis.append("Karies Gigi asimtomatik (Tanpa gejala subjektif signifikan).")
            elif lesi_key in ["ulkus traumatikus", "cheek biting"]:
                if sev >= 5:
                    hasil_sintesis.append(f"Ulserasi Traumatik akut (Nyeri {sev}/10). Berikan obat topikal & eliminasi etiologi.")
                else:
                    hasil_sintesis.append(f"Lesi traumatik tahap penyembuhan / ringan.")
            elif lesi_key == "stain calculus":
                if "berdarah" in char:
                    hasil_sintesis.append("Suspek Periodontitis/Gingivitis (Kalkulus + keluhan perdarahan gingiva).")
                else:
                    hasil_sintesis.append("Kalkulus tanpa komplikasi inflamasi akut yang dilaporkan.")
            else:
                hasil_sintesis.append(f"Terkonfirmasi: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            hasil_sintesis.append(f"Deteksi visual: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
    return " | ".join(hasil_sintesis)

def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

# ============================================================
# HALAMAN LOGIN (SPLIT-SCREEN)
# ============================================================
if not st.session_state.logged_in:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; margin: 0; padding: 0; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        [data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ padding: 0 !important; max-width: 100% !important; }}
        
        .login-bg-left {{
            position: fixed; top: 0; left: 0; width: 50%; height: 100vh;
            background: linear-gradient(145deg, {C_BLUE} 0%, #3B4CCA 100%); z-index: 0;
        }}
        .login-bg-right {{
            position: fixed; top: 0; left: 50%; width: 50%; height: 100vh;
            background: #FFFFFF; z-index: 0;
        }}
        div[data-testid="stForm"] {{ border: none; background: transparent; padding: 0; box-shadow: none; }}
        .stButton>button {{
            background: #111827 !important; color: white !important; border-radius: 8px !important;
            padding: 12px 24px !important; width: 100% !important; font-weight: 700 !important; border: none !important;
        }}
        .stButton>button:hover {{ background: #374151 !important; transform: translateY(-2px); }}
        .stTextInput input {{
            border: none !important; border-bottom: 2px solid #E5E7EB !important; border-radius: 0 !important;
            padding-left: 0 !important; background-color: transparent !important;
        }}
        .stTextInput input:focus {{ border-bottom: 2px solid {C_BLUE} !important; }}
        </style>
        <div class="login-bg-left"></div>
        <div class="login-bg-right"></div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"""
            <div style="height: 100vh; display: flex; flex-direction: column; justify-content: center; padding: 15%; z-index: 1; position: relative;">
                <div style="font-size: 80px; color: white; line-height: 1; margin-bottom: 20px;">*</div>
                <h1 style="color: white !important; font-size: 3.5rem; font-weight: 800; line-height: 1.1; margin-bottom: 0;">Hello</h1>
                <h1 style="color: white !important; font-size: 3.5rem; font-weight: 800; line-height: 1.1; margin-top: 0;">RSGM Unjani! 👋</h1>
                <p style="color: rgba(255,255,255,0.85); font-size: 1.1rem; margin-top: 20px; line-height: 1.6; max-width: 80%;">
                    Skip repetitive and manual medical tasks. Get highly productive through AI automation and save tons of time!
                </p>
                <div style="position: absolute; bottom: 40px; color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                    © 2026 Klinik AI RSGM. All rights reserved.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("<div style='height: 25vh;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="padding: 0 20%; z-index: 1; position: relative;">
                <h2 style="color: #111827 !important; font-weight: 800; font-size: 2rem;">Welcome Back!</h2>
                <p style="color: #6B7280; font-size: 0.9rem; margin-bottom: 40px;">
                    Don't have an account? <span style="text-decoration: underline; color: #111827; font-weight: 600;">Request access now</span>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        _, col_form, _ = st.columns([1, 3, 1])
        with col_form:
            with st.form("login_form"):
                user_input = st.text_input("Email / ID Operator", placeholder="contoh: adinara savero")
                pass_input = st.text_input("Password", type="password", placeholder="••••••••")
                st.markdown("<br>", unsafe_allow_html=True)
                submit_btn = st.form_submit_button("Login Now")
                if submit_btn:
                    if user_input.lower().strip() == "adinara savero" and pass_input == "2560171013":
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("Kredensial tidak valid.")
    st.stop()

# ============================================================
# CSS UTAMA & SIDEBAR (SETELAH LOGIN)
# ============================================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
    .stApp {{ background-color: {BG_MAIN}; }}
    @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .anim-slide {{ animation: slideUp 0.5s ease-out forwards; }}
    h1, h2, h3, h4, h5 {{ color: {TEXT_DARK} !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
    p, label {{ color: {TEXT_MUTED}; }}
    
    section[data-testid="stSidebar"] {{ background-color: {BG_SIDEBAR} !important; border-right: none !important; }}
    section[data-testid="stSidebar"] * {{ color: #94A3B8 !important; }} 
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
    
    div.row-widget.stRadio > div {{ gap: 8px; }}
    div.row-widget.stRadio > div > label {{
        background-color: transparent; padding: 12px 16px; border-radius: 8px; transition: all 0.2s ease; margin-bottom: 2px; cursor: pointer;
    }}
    div.row-widget.stRadio > div > label:hover {{ background-color: rgba(255,255,255,0.05); }}
    div.row-widget.stRadio > div > label p {{ font-weight: 500 !important; font-size: 0.95rem !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{ background-color: {C_BLUE} !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] p {{ color: #FFFFFF !important; font-weight: 600 !important; }}
    
    .glass-card {{
        background: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 16px; padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 24px;
    }}
    .stButton>button {{
        background-color: {C_BLUE} !important; color: white !important; border-radius: 8px !important;
        border: none !important; padding: 14px 24px !important; font-weight: 600 !important; transition: all 0.2s !important;
    }}
    .stButton>button:hover {{ background-color: {C_PURPLE} !important; transform: translateY(-2px); }}
    .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-block; }}
    .badge-low {{ background-color: #E0F2FE; color: #0284C7; }}
    .badge-med {{ background-color: #FEF3C7; color: #D97706; }}
    .badge-high {{ background-color: #FEE2E2; color: #DC2626; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    logo_base64 = get_image_as_base64(LOGO_PATH)
    if logo_base64:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 30px; margin-top: 10px; padding: 0 10px;">
                <img src="data:image/jpeg;base64,{logo_base64}" style="width: 45px; height: 45px; border-radius: 8px; object-fit: contain; background: white; padding: 2px;">
                <div>
                    <h3 style="margin:0; font-size: 1.1rem; color: white !important; font-weight: 700;">{CLINIC_NAME}</h3>
                    <p style="margin:0; font-size: 0.75rem; color: {C_CYAN} !important;">AI Medical Dashboard</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: white !important; margin-bottom: 30px; padding-left: 10px;'>{CLINIC_NAME}</h3>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; margin-left: 10px; margin-bottom: 10px;'>OVERVIEW</p>", unsafe_allow_html=True)
    menu = st.radio("Navigasi", ["Dashboard Skrining", "Rekam Medis (EMR)", "Analitik Kinerja", "Referensi Klinis"], label_visibility="collapsed")
    
    st.markdown("<br><p style='font-size: 0.75rem; font-weight: 700; letter-spacing: 1px; margin-left: 10px; margin-bottom: 10px;'>AI SETTINGS</p>", unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 40px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 35px; height: 35px; border-radius: 50%; background: {C_PURPLE}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">AS</div>
                <div>
                    <p style="margin: 0; color: white !important; font-weight: 600; font-size: 0.9rem; line-height: 1.2;">Adinara Savero</p>
                    <p style="margin: 0; color: {C_CYAN} !important; font-size: 0.75rem;">{USER_ROLE}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Log out", key="logout_btn"):
        st.session_state.logged_in = False
        st.rerun()

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN 1: DASHBOARD
# ============================================================
if menu == "Dashboard Skrining":
    st.markdown("<div class='anim-slide'>", unsafe_allow_html=True)
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(f"<h1 style='margin-bottom: 5px; color: {TEXT_DARK};'>Skrining Lesi Oral</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.05rem; margin-top: 0;'>Sintesis keluhan subyektif (OLD CARTS) dan inferensi visual YOLO.</p>", unsafe_allow_html=True)
    with col_hdr2:
        st.markdown(f"<div style='text-align: right; margin-top: 15px;'><span style='background: white; border: 1px solid {BORDER_COLOR}; padding: 8px 16px; border-radius: 20px; font-weight: 600; color: {C_BLUE};'>{datetime.now().strftime('%d %b %Y')}</span></div>", unsafe_allow_html=True)

    # --- ANAMNESIS ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Fase 1: Anamnesis (OLD CARTS)</h3>", unsafe_allow_html=True)
    with st.form("form_old_carts"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            o_val = st.text_input("Onset (O)", placeholder="Kapan gejala mulai muncul?")
            l_val = st.text_input("Location (L)", placeholder="Di sebelah mana? Menjalar?")
            d_val = st.text_input("Duration (D)", placeholder="Berapa lama gejala bertahan?")
            c_val = st.selectbox("Character (C)", ["", "Berdenyut tajam", "Ngilu/Tumpul", "Terbakar/Panas", "Gatal", "Berdarah", "Lainnya"])
        with c2:
            a_val = st.text_input("Aggravating (A)", placeholder="Apa yang memperburuk?")
            r_val = st.text_input("Relieving (R)", placeholder="Apa yang meredakan?")
            t_val = st.selectbox("Timing (T)", ["", "Sepanjang waktu", "Malam hari saja", "Pagi hari", "Saat/setelah makan"])
            s_val = st.slider("Severity (S) - Visual Analog Scale", 0, 10, 0)
        
        if st.form_submit_button("Simpan Parameter Anamnesis", use_container_width=True):
            st.session_state.anamnesis_data = {
                "O_Onset": o_val, "L_Location": l_val, "D_Duration": d_val, "C_Character": c_val,
                "A_Aggravating": a_val, "R_Relieving": r_val, "T_Timing": t_val, "S_Severity": s_val
            }
            st.success("Parameter berhasil direkam. Silakan lanjutkan ke fase pemindaian visual.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- DETEKSI VISUAL ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Fase 2: Akuisisi Visual</h3>", unsafe_allow_html=True)
    tabs = st.tabs(["Unggah Citra Medis (Batch)", "Kamera Perangkat"])
    images_to_process, file_names = [], []

    with tabs[0]:
        uploaded_files = st.file_uploader("Upload berkas JPG/PNG", accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    images_to_process.append(Image.open(f).convert("RGB"))
                    file_names.append(f.name)
                except Exception: pass

    with tabs[1]:
        camera_file = st.camera_input("Ambil foto klinis")
        if camera_file is not None:
            try:
                images_to_process.append(Image.open(camera_file).convert("RGB"))
                file_names.append(f"Cam_{datetime.now().strftime('%H%M%S')}.jpg")
            except Exception: pass

    if images_to_process:
        st.markdown("---")
        if st.button("🚀 Jalankan Analisis AI & Sintesis Diagnosis", use_container_width=True, disabled=(model is None)):
            progress_bar = st.progress(0, text="Menginisialisasi analisis...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress_bar.progress((idx + 1) / len(images_to_process), text=f"Memproses {f_name}...")
                results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                detections = []
                anamnesis = st.session_state.anamnesis_data or {}
                
                for box in boxes:
                    c_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    n_lesi = model.names[c_id]
                    detections.append((n_lesi, conf))
                
                sintesis_akhir = synthesize_clinical_diagnosis(detections, anamnesis)
                
                record_base = {
                    "ID": str(uuid.uuid4())[:8],
                    "Tanggal": datetime.now().strftime("%Y-%m-%d"),
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Nama_File": f_name,
                    **anamnesis,
                    "Suspek_Diagnosis": sintesis_akhir
                }

                if not detections:
                    all_new_records.append({**record_base, "Lesi_Terdeteksi": "Tidak terdeteksi", "Confidence": 0.0})
                else:
                    for n_lesi, conf in detections:
                        all_new_records.append({**record_base, "Lesi_Terdeteksi": n_lesi, "Confidence": round(conf, 4)})

                st.markdown(f"<h4 style='margin-top: 30px; padding-top: 20px; border-top: 1px solid {BORDER_COLOR};'>Dokumen EMR: {f_name}</h4>", unsafe_allow_html=True)
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;'>Foto Klinis Asli</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_img2:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;'>Deteksi Obyek (YOLO)</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                st.markdown(f"""
                    <div class="anim-slide" style="background: white; border: 1px solid {BORDER_COLOR}; border-left: 6px solid {C_BLUE}; border-radius: 12px; padding: 24px; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                        <p style="font-size: 0.8rem; text-transform: uppercase; color: {TEXT_MUTED}; font-weight: 700; margin: 0 0 10px 0; letter-spacing: 0.05em;">Sintesis Suspek Diagnosis (AI + OLD CARTS)</p>
                        <p style="font-size: 1.1rem; color: {TEXT_DARK}; font-weight: 600; margin: 0; line-height: 1.5;">{sintesis_akhir}</p>
                    </div>
                """, unsafe_allow_html=True)

            progress_bar.progress(1.0, text="Data berhasil direkam ke dalam sistem.")
            append_log(all_new_records)
            st.session_state.anamnesis_data = None
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 2: REKAM MEDIS (EMR & DOWNLOAD LENGKAP)
# ============================================================
elif menu == "Rekam Medis (EMR)":
    st.markdown("<div class='anim-slide'>", unsafe_allow_html=True)
    st.markdown("## Electronic Medical Record (EMR)")
    st.markdown("Arsip lengkap yang memuat Tanggal, ID, Waktu Deteksi, Anamnesis (OLD CARTS), Suspek Diagnosis, dan parameter klinis.")
    
    df_log = load_log()
    if df_log.empty:
        st.info("Basis data rekam medis kosong.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            opts = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Filter Anomali", opts, default=opts)
        with col_f2:
            tgl_s = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            date_range = st.date_input("Filter Waktu Pemeriksaan", value=(tgl_s.min(), tgl_s.max())) if not tgl_s.empty else None
        with col_f3:
            min_conf = st.slider("Filter Confidence Minimum", 0.0, 1.0, 0.0)
        st.markdown("</div>", unsafe_allow_html=True)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= min_conf)
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)

        df_filtered = df_log[mask]
        
        # Tampilkan tabel EMR lengkap
        st.dataframe(df_filtered, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Semua Data (CSV Lengkap)", data=csv, file_name="EMR_Lengkap_RSGM_Unjani.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("📥 Unduh Semua Data (Excel Lengkap)", data=buffer.getvalue(), file_name="EMR_Lengkap_RSGM_Unjani.xlsx", use_container_width=True)
            except ImportError: pass
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 3: ANALITIK KINERJA & GRAFIK PREVALENSI PENYAKIT
# ============================================================
elif menu == "Analitik Kinerja":
    st.markdown("<div class='anim-slide'>", unsafe_allow_html=True)
    st.markdown("## Analitik & Grafik Prevalensi Penyakit")
    st.markdown("Distribusi statistik dan prevalensi temuan patologi oral berdasarkan data rekam medis.")
    
    df_log = load_log()
    
    if df_log.empty:
        st.info("Memerlukan minimal 1 data entri untuk menghasilkan grafis analitik.")
    else:
        tot = len(df_log)
        avg = df_log["Confidence"].mean() * 100
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; padding: 30px;'>
                <div style='font-size: 0.85rem; color: {TEXT_MUTED}; font-weight: 700; letter-spacing: 1px;'>TOTAL KASUS TERCATAT</div>
                <div style='font-size: 3rem; font-weight: 800; color: {C_BLUE}; margin: 10px 0;'>{tot}</div>
                <div style='color: {TEXT_MUTED}; font-size: 0.85rem;'>Database EMR RSGM Unjani</div>
            </div>""", unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; padding: 30px;'>
                <div style='font-size: 0.85rem; color: {TEXT_MUTED}; font-weight: 700; letter-spacing: 1px;'>RATA-RATA PRESISI AI</div>
                <div style='font-size: 3rem; font-weight: 800; color: {C_PURPLE}; margin: 10px 0;'>{avg:.1f}%</div>
                <div style='color: {TEXT_MUTED}; font-size: 0.85rem;'>Tingkat Confidence Model</div>
            </div>""", unsafe_allow_html=True)

        # GRAFIK PREVALENSI PENYAKIT/LESI
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>📊 Grafik Prevalensi Penyakit / Lesi Oral</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem;'>Frekuensi kemunculan masing-masing jenis anomali atau lesi yang terdeteksi dalam sistem.</p>", unsafe_allow_html=True)
        
        # Filter out 'Tidak terdeteksi' untuk grafik prevalensi yang bersih
        df_valid_lesion = df_log[df_log["Lesi_Terdeteksi"] != "Tidak terdeteksi"]
        if not df_valid_lesion.empty:
            prevalensi_counts = df_valid_lesion["Lesi_Terdeteksi"].value_counts()
            st.bar_chart(prevalensi_counts, color=C_BLUE)
        else:
            st.info("Belum ada data lesi positif yang terekam untuk grafik prevalensi.")
        st.markdown("</div>", unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Distribusi Tingkat Nyeri (Severity)</h4>", unsafe_allow_html=True)
            if "S_Severity" in df_log.columns and not df_log["S_Severity"].dropna().empty:
                st.bar_chart(df_log["S_Severity"].value_counts().sort_index(), color=C_PURPLE)
            else:
                st.info("Data tingkat nyeri belum tersedia.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_c2:
            st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Tren Skrining Kumulatif Harian</h4>", unsafe_allow_html=True)
            st.line_chart(df_log.groupby("Tanggal").size(), color=C_CYAN)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 4: REFERENSI ENSIKLOPEDIA
# ============================================================
elif menu == "Referensi Klinis":
    st.markdown("<div class='anim-slide'>", unsafe_allow_html=True)
    st.markdown("## Ensiklopedia Patologi Oral")
    st.markdown("Standar referensi dan panduan tindakan berbasis *Evidence-Based Dentistry*.")
    
    for key, info in LESION_INFO.items():
        urg = info['urgensi']
        bg_badge = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
        
        st.markdown(f"""
            <div class="glass-card" style="padding: 24px; margin-bottom: 20px; border-left: 5px solid {C_BLUE};">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 16px; margin-bottom: 16px;">
                    <h3 style="margin: 0; color: {TEXT_DARK} !important;">{info['nama_klinis']}</h3>
                    <span class="badge {bg_badge}">{urg}</span>
                </div>
                <p style="color: {TEXT_DARK}; font-size: 1rem; margin-bottom: 16px; font-weight: 500;">{info['deskripsi']}</p>
                <div style="background: {BG_MAIN}; padding: 16px; border-radius: 8px; font-size: 0.9rem; color: {C_BLUE}; font-weight: 600;">
                    Rekomendasi Tindakan: <span style="font-weight: 400; color: {TEXT_DARK}">{info['rekomendasi']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
