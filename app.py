"""
Klinik AI RSGM — Sistem Skrining Lesi Oral berbasis YOLO
=========================================================
Antarmuka: Frost UI, Clean, dan Modern SaaS
Fitur Baru: Login, OLD CARTS, Batch Upload, EMR Export, AI Synthesis
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
APP_VERSION = "3.0 (Frost UI & OLD CARTS Integration)"
CLINIC_NAME = "RSGM Unjani"
USER_NAME = "drg. Adinara Savero, S.KG"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

# Kredensial Login
OPERATOR_ID = "adinara savero"
OPERATOR_PASS = "2560171013"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi.csv")

# Penambahan kolom OLD CARTS dan Suspek Diagnosis tanpa menghapus kolom lama
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi", "Confidence", "Model_Version", "Nama_File",
    "O_Onset", "L_Location", "D_Duration", "C_Character", 
    "A_Aggravating", "R_Relieving", "T_Timing", "S_Severity", "Suspek_Diagnosis"
]

# PALET WARNA (Modern Frost UI Medical SaaS)
PRIMARY = "#0f766e"        # Deep Teal
PRIMARY_LIGHT = "#ccfbf1"
ACCENT = "#fbbf24"         # Amber
DANGER = "#ef4444"         # Red
DARK_TEXT = "#0f172a"      # Slate 900
GRAY_TEXT = "#64748b"      # Slate 500
BG_MIST = "#f4f7f9"        # Frost background
BORDER_COLOR = "#e2e8f0"

LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum (Cheek Biting)",
        "deskripsi": "Morsicatio buccarum mengacu pada kebiasaan menggigit atau menggerogoti mukosa bukal, yang seringkali mengakibatkan tampilan kasar dan bergerigi pada area yang terkena dan berpotensi menyebabkan ulserasi atau erosi. Kondisi ini mungkin terkait dengan stres atau gangguan psikologis., umumnya tampak sebagai area putih ireguler.",
        "rekomendasi": "Edukasi pasien untuk menghentikan kebiasaan menggigit pipi; evaluasi ulang bila lesi menetap lebih dari 2 minggu.",
        "urgensi": "Rendah",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue (Lidah Berlapis)",
        "deskripsi": "Kondisi klinis yang terjadi pada bagian permukaan lidah yang ditutupi oleh suatu selaput pseudomembran yang terjadi akibat penumpukan debris atau sisa makanan, sel-sel keratin yang tidak terdeskuamasi, dan dapat ditemukan adanya mikroorganisme seperti bakteri maupun jamur.",
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
        "nama_klinis": "Torus",
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

# --- INISIALISASI SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'anamnesis_data' not in st.session_state:
    st.session_state.anamnesis_data = None

st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# FUNGSI BANTU & LOGIKA AI
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

def synthesize_clinical_diagnosis(detections, anamnesis):
    """Menghasilkan suspek diagnosis cerdas berdasarkan visual YOLO dan teks OLD CARTS."""
    if not detections:
        return "Tidak terdeteksi anomali visual. Pertimbangkan observasi berbasis keluhan (OLD CARTS)."
    hasil = []
    for nama_lesi, _ in detections:
        key = nama_lesi.lower()
        if anamnesis:
            sev = int(anamnesis.get("S_Severity", 0))
            char = str(anamnesis.get("C_Character", "")).lower()
            
            if key == "karies":
                if sev >= 6 or "denyut" in char:
                    hasil.append("Suspek Pulpitis Irreversibel (Karies disertai nyeri tajam/berdenyut).")
                elif sev >= 3 or "ngilu" in char:
                    hasil.append("Suspek Pulpitis Reversibel (Karies dengan sensitivitas ringan/ngilu).")
                else:
                    hasil.append("Karies Asimtomatik (Tanpa keluhan subjektif signifikan).")
            elif key in ["ulkus traumatikus", "cheek biting"]:
                if sev >= 5:
                    hasil.append(f"Lesi traumatik reaktif akut (Tingkat nyeri {sev}/10). Butuh intervensi topikal.")
                else:
                    hasil.append("Lesi traumatik fase penyembuhan / asimtomatik.")
            else:
                hasil.append(f"Konfirmasi visual: {LESION_INFO.get(key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            hasil.append(f"Deteksi AI: {LESION_INFO.get(key, {}).get('nama_klinis', nama_lesi)}.")
    return " | ".join(hasil)


# ============================================================
# HALAMAN LOGIN (FROST UI)
# ============================================================
if not st.session_state.logged_in:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; margin: 0; padding: 0; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        [data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ padding: 0 !important; max-width: 100% !important; }}
        
        .login-wrapper {{
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #e0f2fe 0%, #ccfbf1 100%);
        }}
        .login-card {{
            background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.5); border-radius: 24px;
            padding: 40px; width: 100%; max-width: 450px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
            animation: slideUp 0.6s ease-out forwards;
        }}
        @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        
        .stTextInput input {{
            border: 1px solid #cbd5e1 !important; border-radius: 12px !important; padding: 12px 16px !important;
            transition: all 0.2s !important;
        }}
        .stTextInput input:focus {{ border-color: {PRIMARY} !important; box-shadow: 0 0 0 3px {PRIMARY_LIGHT} !important; }}
        
        .stButton>button {{
            background: {PRIMARY} !important; color: white !important; width: 100% !important;
            border-radius: 12px !important; padding: 12px !important; font-weight: 600 !important;
            transition: all 0.2s !important; border: none !important; margin-top: 10px !important;
        }}
        .stButton>button:hover {{ background: #115e59 !important; transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.1) !important; }}
        div[data-testid="stForm"] {{ border: none; background: transparent; padding: 0; }}
        </style>
        
        <div class="login-wrapper">
            <div class="login-card">
                <div style="text-align: center; margin-bottom: 30px;">
                    <div style="width: 48px; height: 48px; background: {PRIMARY_LIGHT}; color: {PRIMARY}; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold; margin-bottom: 15px;">✦</div>
                    <h2 style="margin: 0; color: {DARK_TEXT}; font-weight: 800; font-size: 1.5rem;">RSGM UNJANI</h2>
                    <p style="margin: 5px 0 0 0; color: {GRAY_TEXT}; font-size: 0.9rem;">Clinical Intelligence Workspace</p>
                </div>
    """, unsafe_allow_html=True)
    
    col_space1, col_login, col_space3 = st.columns([1, 2, 1])
    with col_login:
        with st.form("login_form"):
            st.markdown(f"<label style='font-size: 0.85rem; font-weight: 600; color: {DARK_TEXT};'>Email / ID Operator</label>", unsafe_allow_html=True)
            user_input = st.text_input("ID", label_visibility="collapsed", placeholder="contoh: adinara savero")
            
            st.markdown(f"<br><label style='font-size: 0.85rem; font-weight: 600; color: {DARK_TEXT};'>Password</label>", unsafe_allow_html=True)
            pass_input = st.text_input("Pass", label_visibility="collapsed", type="password", placeholder="••••••••")
            
            submit_btn = st.form_submit_button("LOGIN")
            
            if submit_btn:
                if user_input.lower().strip() == OPERATOR_ID and pass_input == OPERATOR_PASS:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Kredensial tidak valid.")
        
        st.markdown(f"""
            <div style="margin-top: 20px; text-align: center; background: #f8fafc; padding: 12px; border-radius: 12px; border: 1px dashed #cbd5e1;">
                <p style="margin: 0; font-size: 0.8rem; font-weight: 600; color: {PRIMARY};">Akses Demo AI</p>
                <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT};">ID: adinara savero | Pass: 2560171013</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# CSS UTAMA (FROST UI SETELAH LOGIN)
# ============================================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: {BG_MIST}; }}
    
    /* Global Animations */
    @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .anim-fade-up {{ animation: fadeUp 0.5s ease-out forwards; }}
    
    h1, h2, h3, h4, h5, h6 {{ color: {DARK_TEXT} !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
    p, label {{ color: {GRAY_TEXT}; }}
    
    /* FROST UI / GLASSMORPHISM CARDS */
    .frost-card {{
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -1px rgba(15, 23, 42, 0.03);
        margin-bottom: 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .frost-card:hover {{ box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.05); }}
    
    /* BUTTONS */
    .stButton>button {{
        background-color: {PRIMARY} !important; color: white !important;
        border-radius: 10px !important; border: none !important; padding: 12px 24px !important;
        font-weight: 600 !important; transition: all 0.2s ease !important;
    }}
    .stButton>button:hover {{ background-color: #115e59 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(15,23,42,0.1) !important; }}
    
    /* BADGES */
    .badge {{ padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-block; text-transform: uppercase; letter-spacing: 0.05em; }}
    .badge-low {{ background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
    .badge-med {{ background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
    .badge-high {{ background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
    .badge-conf-high {{ background-color: #d1fae5; color: #047857; border: 1px solid #10b981; margin-right: 8px; }}
    .badge-conf-med {{ background-color: #fef3c7; color: #b45309; border: 1px solid #f59e0b; margin-right: 8px; }}
    .badge-conf-low {{ background-color: #fee2e2; color: #b91c1c; border: 1px solid #ef4444; margin-right: 8px; }}
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {{ background-color: rgba(255,255,255,0.85) !important; backdrop-filter: blur(10px); border-right: 1px solid {BORDER_COLOR}; }}
    [data-testid="stSidebar"] * {{ color: {DARK_TEXT} !important; }}
    
    /* NAVIGATION RADIO STYLING */
    div.row-widget.stRadio > div {{ gap: 5px; }}
    div.row-widget.stRadio > div > label {{ background-color: transparent; padding: 12px 16px; border-radius: 10px; transition: all 0.2s ease; cursor: pointer; }}
    div.row-widget.stRadio > div > label:hover {{ background-color: {PRIMARY_LIGHT}; }}
    div.row-widget.stRadio > div > label p {{ font-weight: 500 !important; font-size: 0.95rem !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{ background-color: {PRIMARY} !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] p {{ color: #FFFFFF !important; font-weight: 600 !important; }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="margin-bottom: 32px; display: flex; align-items: center; gap: 10px;">
            <div style="width: 36px; height: 36px; background: {PRIMARY}; color: white; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;">✦</div>
            <div>
                <h2 style="margin:0; font-size: 1.2rem; color: {DARK_TEXT} !important; font-weight: 800;">{CLINIC_NAME}</h2>
                <p style="margin:0; font-size: 0.75rem; color: {GRAY_TEXT} !important; font-weight: 500;">AI Vision Screening</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", ["Dashboard", "Riwayat Deteksi", "Analitik", "Referensi Lesi", "Sistem"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 5px;'>Pengaturan Model</p>", unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU (NMS) Threshold", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 40px; padding: 16px; background: white; border-radius: 12px; border: 1px solid {BORDER_COLOR}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <p style="margin: 0; font-size: 0.7rem; color: {GRAY_TEXT}; text-transform: uppercase; font-weight: 700;">Operator Aktif</p>
            <p style="margin: 4px 0 0 0; font-weight: 700; font-size: 0.9rem; color: {DARK_TEXT};">{USER_NAME}</p>
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT};">{USER_ROLE}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Log Out", key="logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.anamnesis_data = None
        st.rerun()

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard":
    st.markdown("<div class='anim-fade-up'>", unsafe_allow_html=True)
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"<p style='color: {PRIMARY}; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 5px;'>DASHBOARD SKRINING</p>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin-top: 0; margin-bottom: 4px; font-size: 2.2rem;'>Analisis Lesi Intraoral</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; margin-top: 0;'>Deteksi dini dan dokumentasi klinis berbasis Computer Vision & Anamnesis.</p>", unsafe_allow_html=True)
    with col_header2:
        st.markdown(f"<div style='text-align: right; margin-top: 24px; font-weight: 600; color: {GRAY_TEXT}; font-size: 0.9rem; background: white; padding: 8px 16px; border-radius: 20px; border: 1px solid {BORDER_COLOR}; display: inline-block; float: right;'>{datetime.now().strftime('%d %b %Y')}</div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; padding: 16px; border-radius: 12px; margin-bottom: 24px; display: flex; gap: 12px;">
            <span style="font-size: 1.2rem;">⚠️</span>
            <p style="margin: 0; color: #78350f; font-size: 0.95rem; font-weight: 500;">
                <strong>Disclaimer Eksplisit:</strong> Web ini hanya alat bantu skrining. Keputusan diagnosis tetap berada pada dokter .
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- FITUR BARU: ANAMNESIS OLD CARTS ---
    st.markdown("<div class='frost-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 15px;">
            <div style="width: 32px; height: 32px; background: #e0f2fe; color: #0284c7; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem;">📋</div>
            <h3 style="margin: 0; font-size: 1.2rem;">Anamnesis (OLD CARTS)</h3>
        </div>
        <p style="font-size: 0.9rem; margin-bottom: 20px;">Lengkapi data anamnesis sebelum melakukan skrining visual agar AI dapat memberikan pertimbangan suspek diagnosis.</p>
    """, unsafe_allow_html=True)
    
    with st.form("anamnesis_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            o_val = st.text_input("Onset (O)", placeholder="Kapan gejala pertama muncul & sudah berapa lama?")
            l_val = st.text_input("Location (L)", placeholder="Di bagian mana? Apakah menjalar?")
            d_val = st.text_input("Duration (D)", placeholder="Terus-menerus atau hilang timbul?")
            c_val = st.text_input("Character (C)", placeholder="Bentuk rasa sakit (tajam, tumpul, berdenyut)?")
        with c2:
            a_val = st.text_input("Aggravating (A)", placeholder="Apa yang memperparah?")
            r_val = st.text_input("Relieving (R)", placeholder="Apa yang meredakan?")
            t_val = st.text_input("Timing (T)", placeholder="Kapan biasanya memburuk (pagi/malam)?")
            s_val = st.slider("Severity (S) - Skala Nyeri 1-10", 0, 10, 0)
        
        if st.form_submit_button("Simpan Data"):
            st.session_state.anamnesis_data = {
                "O_Onset": o_val or "-", "L_Location": l_val or "-", "D_Duration": d_val or "-", 
                "C_Character": c_val or "-", "A_Aggravating": a_val or "-", "R_Relieving": r_val or "-", 
                "T_Timing": t_val or "-", "S_Severity": s_val
            }
            st.success("Anamnesis berhasil direkam. Silakan unggah foto klinis di bawah.")
    st.markdown("</div>", unsafe_allow_html=True)


    # --- AKUISISI VISUAL BATCH ---
    st.markdown("<div class='frost-card'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 15px;">
            <div style="width: 32px; height: 32px; background: #fce7f3; color: #be185d; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem;">📸</div>
            <h3 style="margin: 0; font-size: 1.2rem;">Deteksi Foto Klinis</h3>
        </div>
    """, unsafe_allow_html=True)
    
    tab_unggah, tab_kamera = st.tabs(["Unggah Berkas (Batch)", "Kamera Perangkat"])
    images_to_process, file_names = [], []

    with tab_unggah:
        uploaded_files = st.file_uploader("Unggah banyak gambar (multi-gambar) sekaligus", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
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
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Mulai Proses Inferensi YOLO & Sintesis", use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            if not st.session_state.anamnesis_data:
                st.warning("Anda belum menyimpan data OLD CARTS. Analisis akan berjalan tanpa konteks anamnesis.")
                anamnesis = {}
            else:
                anamnesis = st.session_state.anamnesis_data

            progress = st.progress(0, text="Memulai pemrosesan batch...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress.progress((idx + 1) / len(images_to_process), text=f"Menganalisis {f_name}...")
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
                
                # SINTESIS PINTAR
                sintesis = synthesize_clinical_diagnosis(detections, anamnesis)
                
                base_record = {
                    "ID": str(uuid.uuid4())[:8].upper(),
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Tanggal": datetime.now().strftime("%Y-%m-%d"),
                    "Model_Version": MODEL_PATH.name,
                    "Nama_File": f_name,
                    **anamnesis,
                    "Suspek_Diagnosis": sintesis
                }

                if not detections:
                    all_new_records.append({**base_record, "Lesi_Terdeteksi": "Tidak terdeteksi", "Confidence": 0.0})
                else:
                    for nama_lesi, conf_score in detections:
                        all_new_records.append({**base_record, "Lesi_Terdeteksi": nama_lesi, "Confidence": round(conf_score, 4)})

                # RENDER HASIL
                st.markdown(f"""
                    <div style="background: white; border: 1px solid {BORDER_COLOR}; border-radius: 16px; padding: 24px; margin-top: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <h4 style="margin-top: 0; font-size: 1.1rem; border-bottom: 1px dashed {BORDER_COLOR}; padding-bottom: 12px; margin-bottom: 16px;">Dokumen: {f_name}</h4>
                """, unsafe_allow_html=True)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #64748b;'>Citra Asli</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_img2:
                    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #64748b;'>Overlay YOLO</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                if not detections:
                    st.info("Tidak terdeteksi adanya anomali visual berdasarkan parameter.")
                else:
                    st.markdown("<h5 style='margin-top: 20px; font-size: 1rem;'>Kartu Informasi Klinis</h5>", unsafe_allow_html=True)
                    for nama_lesi, conf_score in sorted(detections, key=lambda x: -x[1]):
                        info = get_lesion_info(nama_lesi)
                        urg = info['urgensi']
                        badge_urg = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
                        
                        conf_pct = conf_score * 100
                        badge_conf = "badge-conf-high" if conf_pct >= 75 else "badge-conf-med" if conf_pct >= 50 else "badge-conf-low"
                        
                        st.markdown(f"""
                            <div style="background: {BG_MIST}; border-radius: 12px; padding: 20px; margin-bottom: 12px; border: 1px solid {BORDER_COLOR}; border-left: 4px solid {PRIMARY};">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="font-weight: 700; font-size: 1.05rem; color: {DARK_TEXT};">{info['nama_klinis']}</span>
                                    <div>
                                        <span class="badge {badge_conf}">Conf: {conf_pct:.1f}%</span>
                                        <span class="badge {badge_urg}">Urgensi: {info['urgensi']}</span>
                                    </div>
                                </div>
                                <p style="font-size: 0.9rem; color: {GRAY_TEXT}; margin-bottom: 12px; line-height: 1.5;">{info['deskripsi']}</p>
                                <div style="background: white; border-radius: 8px; padding: 12px; border: 1px solid {BORDER_COLOR}; font-size: 0.85rem; font-weight: 500;">
                                    <strong>Rekomendasi:</strong> {info['rekomendasi']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # KARTU SINTESIS
                st.markdown(f"""
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 16px; margin-top: 15px;">
                        <p style="margin: 0 0 5px 0; font-size: 0.75rem; font-weight: 700; color: #166534; text-transform: uppercase;">Pertimbangan Suspek Diagnosis (AI + OLD CARTS)</p>
                        <p style="margin: 0; font-size: 0.95rem; font-weight: 600; color: #14532d;">{sintesis}</p>
                    </div>
                    </div>
                """, unsafe_allow_html=True)

            progress.progress(1.0, text="Analisis Batch Selesai. Data tersimpan di Riwayat.")
            append_log(all_new_records)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HALAMAN: RIWAYAT DETEKSI
# ============================================================
elif menu == "Riwayat Deteksi":
    st.markdown("<div class='anim-fade-up'>", unsafe_allow_html=True)
    st.markdown("## Database Pemeriksaan (EMR)")
    st.markdown("Arsip historis deteksi klinis dan OLD CARTS yang terekam.")
    df_log = load_log()

    if df_log.empty:
        st.info("Basis data log saat ini kosong.")
    else:
        st.markdown("<div class='frost-card'>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            opts = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Filter Jenis Lesi", opts, default=opts)
        with col_f2:
            tgl_series = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            date_range = st.date_input("Rentang Waktu", value=(tgl_series.min(), tgl_series.max())) if not tgl_series.empty else None
        with col_f3:
            min_conf = st.slider("Batas Minimum Confidence (%)", 0, 100, 0)
        st.markdown("</div>", unsafe_allow_html=True)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= (min_conf/100.0))
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)

        df_filtered = df_log[mask]
        
        st.markdown("<div class='frost-card' style='padding: 10px;'>", unsafe_allow_html=True)
        st.dataframe(df_filtered, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Ekspor Format CSV", data=df_filtered.to_csv(index=False).encode("utf-8"), file_name="Riwayat_Klinis.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("📥 Ekspor Format Excel", data=buffer.getvalue(), file_name="Riwayat_Klinis.xlsx", use_container_width=True)
            except ImportError: pass
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HALAMAN: ANALITIK
# ============================================================
elif menu == "Analitik":
    st.markdown("<div class='anim-fade-up'>", unsafe_allow_html=True)
    st.markdown("## Analitik & Statistik")
    st.markdown("Ringkasan distribusi, prevalensi, dan metrik kinerja.")
    df_log = load_log()
    
    if df_log.empty: 
        st.info("Tidak ada data analitik tersedia.")
    else:
        tot = len(df_log)
        avg_conf = df_log['Confidence'].mean() * 100
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class='frost-card' style='text-align: center;'>
                    <p style='margin:0; font-size: 0.8rem; font-weight: 700; color:{GRAY_TEXT}; text-transform: uppercase;'>Total Deteksi</p>
                    <p style='margin:10px 0 0 0; font-size: 2.5rem; font-weight: 800; color:{PRIMARY};'>{tot}</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class='frost-card' style='text-align: center;'>
                    <p style='margin:0; font-size: 0.8rem; font-weight: 700; color:{GRAY_TEXT}; text-transform: uppercase;'>Avg Confidence</p>
                    <p style='margin:10px 0 0 0; font-size: 2.5rem; font-weight: 800; color:#0284c7;'>{avg_conf:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            df_valid = df_log[df_log["Lesi_Terdeteksi"] != "Tidak terdeteksi"]
            top_lesion = df_valid["Lesi_Terdeteksi"].mode()[0] if not df_valid.empty else "N/A"
            st.markdown(f"""
                <div class='frost-card' style='text-align: center;'>
                    <p style='margin:0; font-size: 0.8rem; font-weight: 700; color:{GRAY_TEXT}; text-transform: uppercase;'>Prevalensi Tertinggi</p>
                    <p style='margin:15px 0 0 0; font-size: 1.2rem; font-weight: 800; color:#b45309;'>{top_lesion.title()}</p>
                </div>
            """, unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<div class='frost-card'><h4 style='margin-top:0;'>Distribusi Lesi</h4>", unsafe_allow_html=True)
            if not df_valid.empty: st.bar_chart(df_valid["Lesi_Terdeteksi"].value_counts(), color="#38bdf8")
            else: st.info("Belum ada temuan.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_s2:
            st.markdown("<div class='frost-card'><h4 style='margin-top:0;'>Distribusi Severity (OLD CARTS)</h4>", unsafe_allow_html=True)
            if "S_Severity" in df_log.columns and not df_log["S_Severity"].dropna().empty:
                st.bar_chart(df_log["S_Severity"].value_counts().sort_index(), color="#fbbf24")
            else: st.info("Data anamnesis kosong.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='frost-card'><h4 style='margin-top:0;'>Tren Skrining Harian</h4>", unsafe_allow_html=True)
        st.line_chart(df_log.groupby("Tanggal").size(), color=PRIMARY)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HALAMAN: REFERENSI LESI
# ============================================================
elif menu == "Referensi Lesi":
    st.markdown("<div class='anim-fade-up'>", unsafe_allow_html=True)
    st.markdown("## Ensiklopedia Lesi Oral")
    st.markdown("Rangkuman edukasi singkat tiap kelas lesi pada proyek ini.")
    
    st.markdown(f"""
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 16px; margin-bottom: 24px;">
            <p style="margin: 0; color: #1e3a8a; font-size: 0.9rem; font-weight: 500;">Referensi ini berfungsi sebagai pengingat dasar. Tetap rujuk literatur kedokteran gigi resmi untuk pengambilan keputusan akhir.</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    idx = 0
    for key, info in LESION_INFO.items():
        badge_class = "badge-high" if "Tinggi" in info['urgensi'] else "badge-med" if info['urgensi'] == "Sedang" else "badge-low"
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="frost-card" style="padding: 20px; height: 100%; border-top: 4px solid {PRIMARY};">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <h4 style="margin: 0; font-size: 1.1rem; color: {DARK_TEXT} !important;">{info['nama_klinis']}</h4>
                        <span class="badge {badge_class}" style="font-size: 0.6rem;">{info['urgensi']}</span>
                    </div>
                    <p style="color: {GRAY_TEXT}; font-size: 0.85rem; margin-bottom: 16px; line-height: 1.5;">{info['deskripsi']}</p>
                    <div style="background: {BG_MIST}; padding: 12px; border-radius: 8px; font-size: 0.85rem;">
                        <strong style="color: {PRIMARY};">Panduan Klinis:</strong><br> {info['rekomendasi']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        idx += 1
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HALAMAN: SISTEM
# ============================================================
elif menu == "Sistem":
    st.markdown("<div class='anim-fade-up'>", unsafe_allow_html=True)
    st.markdown("## Informasi Infrastruktur")
    st.markdown(f"""
    <div class='frost-card'>
        <h4 style="margin-top:0;">Spesifikasi Deployment</h4>
        <ul style="color: {GRAY_TEXT}; font-size: 0.95rem; line-height: 1.8;">
            <li><strong>Framework UI:</strong> Streamlit / Python 3 (Frost UI Glassmorphism Mod)</li>
            <li><strong>Architecture AI:</strong> YOLO by Ultralytics</li>
            <li><strong>Data Binding:</strong> Pandas DataFrame & Streamlit Session State</li>
            <li><strong>Environment:</strong> Frontend terintegrasi backend lokal</li>
        </ul>
        <hr style="border-color: {BORDER_COLOR}; margin: 24px 0;">
        <h4>Kontak Dukungan</h4>
        <p style="color: {GRAY_TEXT}; font-size: 0.9rem;">Pembaruan parameter bobot atau manajemen instans dapat dilakukan melalui administrator server utama institusi terkait.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
