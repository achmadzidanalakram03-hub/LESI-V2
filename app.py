"""
Klinik AI RSGM — Sistem Skrining Lesi Oral & Anamnesis Terpadu
=============================================================
Antarmuka: Frost UI (Glassmorphism, Work Sans, Modern Medical SaaS)
Fitur: Login, OLD CARTS, Batch Upload, EMR Filter, Analitik
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
APP_VERSION = "5.0 (Frost UI & OLD CARTS Integration)"
CLINIC_NAME = "RSGM Unjani"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"
OPERATOR_ID = "adinara savero"
OPERATOR_PASS = "2560171013"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi_emr.csv")

DB_COLUMNS = [
    "ID", "Tanggal", "Waktu", "Nama_File", "Lesi_Terdeteksi", "Confidence",
    "O_Onset", "L_Location", "D_Duration", "C_Character", 
    "A_Aggravating", "R_Relieving", "T_Timing", "S_Severity", "Suspek_Diagnosis"
]

# PALET WARNA (Frost UI & Modern Medical)
C_VIOLET = "#A163F7"
C_BLUE   = "#6F88FC"
C_CYAN   = "#45E3FF"
C_INK    = "#1E293B"
C_MUTED  = "#64748B"
C_PAPER  = "#F4F7F9"
C_SIDE   = "#1A1D2D"
BORDER_COLOR = "#e8edf2"

LESION_INFO = {
    "cheek biting": {"nama_klinis": "Morsicatio Buccarum", "deskripsi": "Lesi traumatik akibat gigitan berulang pada mukosa pipi.", "rekomendasi": "Edukasi hilangkan habit; evaluasi 2 minggu.", "urgensi": "Rendah"},
    "coated tongue": {"nama_klinis": "Coated Tongue", "deskripsi": "Penumpukan debris keratin pada dorsum lidah.", "rekomendasi": "Instruksi pembersihan dengan tongue scraper.", "urgensi": "Rendah"},
    "karies": {"nama_klinis": "Karies Gigi", "deskripsi": "Demineralisasi jaringan keras gigi akibat bakteri.", "rekomendasi": "Rujuk untuk preparasi dan restorasi/perawatan saluran akar.", "urgensi": "Sedang-Tinggi"},
    "linea alba": {"nama_klinis": "Linea Alba Buccalis", "deskripsi": "Garis hiperkeratosis putih sejajar bidang oklusal.", "rekomendasi": "Lesi jinak, tidak perlu intervensi khusus.", "urgensi": "Rendah"},
    "lingual varicosites": {"nama_klinis": "Lingual Varicosities", "deskripsi": "Pelebaran pembuluh darah vena di ventral lidah.", "rekomendasi": "Observasi klinis, fisiologis pada usia lanjut.", "urgensi": "Rendah"},
    "stain calculus": {"nama_klinis": "Stain & Kalkulus", "deskripsi": "Deposit mineral plak terkalsifikasi.", "rekomendasi": "Tindakan scaling dan root planing (SRP).", "urgensi": "Sedang"},
    "torus": {"nama_klinis": "Torus", "deskripsi": "Eksostosis tulang jinak asimtomatik.", "rekomendasi": "Observasi kecuali mengganggu pembuatan protesa.", "urgensi": "Rendah"},
    "ulkus traumatikus": {"nama_klinis": "Ulkus Traumatikus", "deskripsi": "Hilangnya lapisan epitel akibat trauma.", "rekomendasi": "Kendalikan faktor etiologi, berikan obat topikal.", "urgensi": "Sedang"},
    "olp": {"nama_klinis": "Oral Lichen Planus", "deskripsi": "Kondisi peradangan kronis pada mukosa mulut.", "rekomendasi": "Evaluasi klinis lanjutan, pertimbangkan biopsi jika atipikal.", "urgensi": "Tinggi"},
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
    if not DB_FILE.exists() or DB_FILE.stat().st_size == 0:
        pd.DataFrame(columns=DB_COLUMNS).to_csv(DB_FILE, index=False)
        return
    try: df = pd.read_csv(DB_FILE)
    except Exception: df = pd.DataFrame(columns=DB_COLUMNS)
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
    return {"nama_klinis": str(nama).title(), "deskripsi": "Informasi klinis belum tersedia di database.", "rekomendasi": "Evaluasi klinis mendalam oleh dokter gigi.", "urgensi": "Tidak Diketahui"}

def synthesize_clinical_diagnosis(detections, anamnesis):
    if not detections:
        return "Tidak teridentifikasi kelainan struktural secara visual. Lakukan evaluasi berbasis keluhan subjektif (OLD CARTS)."
    hasil_sintesis = []
    for nama_lesi, _ in detections:
        lesi_key = nama_lesi.lower()
        if anamnesis:
            sev = anamnesis.get("S_Severity", 0)
            char = str(anamnesis.get("C_Character", "")).lower()
            timing = str(anamnesis.get("T_Timing", "")).lower()
            
            if lesi_key == "karies":
                if sev >= 7 or "denyut" in char or "malam" in timing:
                    hasil_sintesis.append("Suspek Pulpitis Irreversibel (Karies profunda + nyeri spontan parah).")
                elif sev >= 4 or "ngilu" in char:
                    hasil_sintesis.append("Suspek Pulpitis Reversibel (Karies terdeteksi + ngilu terpicu stimulus).")
                else:
                    hasil_sintesis.append("Karies asimtomatik (Tanpa keluhan nyeri signifikan).")
            elif lesi_key in ["ulkus traumatikus", "cheek biting", "linea alba"]:
                if sev >= 5:
                    hasil_sintesis.append(f"Lesi Reaktif/Traumatik akut (Nyeri {sev}/10). Berikan obat topikal & eliminasi etiologi.")
                else:
                    hasil_sintesis.append(f"Lesi Reaktif/Traumatik asimtomatik atau tahap penyembuhan.")
            elif lesi_key == "stain calculus":
                if "berdarah" in char:
                    hasil_sintesis.append("Suspek Periodontitis/Gingivitis (Kalkulus + keluhan perdarahan).")
                else:
                    hasil_sintesis.append("Stain/Kalkulus tanpa komplikasi inflamasi akut yang dilaporkan.")
            else:
                hasil_sintesis.append(f"Terkonfirmasi Visual: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            hasil_sintesis.append(f"Skrining Visual AI: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
    return " | ".join(hasil_sintesis)


# ============================================================
# HALAMAN LOGIN (SPLIT-SCREEN FROST UI)
# ============================================================
if not st.session_state.logged_in:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Work Sans', sans-serif !important; margin: 0; padding: 0; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        [data-testid="stSidebar"] {{ display: none; }}
        .block-container {{ padding: 0 !important; max-width: 100% !important; }}
        
        /* Animasi */
        @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: none; }} }}
        .fade-up {{ animation: fadeUp 0.6s ease both; }}
        
        .login-bg-left {{
            position: fixed; top: 0; left: 0; width: 50%; height: 100vh;
            background: linear-gradient(145deg, #5477ed 0%, #8d65ee 58%, #45dff7 145%); z-index: 0; overflow: hidden;
        }}
        .login-bg-left::before, .login-bg-left::after {{
            content: ""; position: absolute; border: 1px solid rgba(255,255,255,0.23); border-radius: 999px;
        }}
        .login-bg-left::before {{ width: 430px; height: 430px; right: -170px; top: -160px; }}
        .login-bg-left::after {{ width: 300px; height: 300px; left: -155px; bottom: -130px; }}
        
        .login-bg-right {{ position: fixed; top: 0; left: 50%; width: 50%; height: 100vh; background: #FFFFFF; z-index: 0; }}
        
        .stTextInput input {{
            border: 1px solid #dbe3ec !important; border-radius: 10px !important; padding: 12px 14px !important;
            box-shadow: none !important; transition: 0.2s !important;
        }}
        .stTextInput input:focus {{ border-color: {C_VIOLET} !important; box-shadow: 0 0 0 3px rgba(161,99,247,0.13) !important; }}
        
        .stButton>button {{
            background: linear-gradient(135deg, #8768f5, #5d83fa) !important; color: white !important;
            border-radius: 12px !important; padding: 14px 24px !important; font-weight: 700 !important;
            border: none !important; width: 100%; box-shadow: 0 8px 18px rgba(111,136,252,0.24) !important;
            transition: 0.2s !important;
        }}
        .stButton>button:hover {{ transform: translateY(-2px); filter: brightness(1.05); }}
        
        div[data-testid="stForm"] {{ border: none; background: transparent; padding: 0; }}
        </style>
        
        <div class="login-bg-left">
            <div class="fade-up" style="padding: 12%; color: white; display: flex; flex-direction: column; justify-content: center; height: 100%;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 40px;">
                    <div style="width: 40px; height: 40px; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: bold;">✦</div>
                    <span style="font-weight: 700; letter-spacing: 1px;">RSGM UNJANI</span>
                </div>
                <p style="text-transform: uppercase; letter-spacing: 0.22em; font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;">Sistem Klinis Terintegrasi</p>
                <h1 style="font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 0; color: white !important;">Hello RSGM Unjani!</h1>
                <p style="font-size: 1.1rem; color: rgba(255,255,255,0.85); line-height: 1.6; margin-top: 20px;">Satu ruang kerja untuk menyatukan anamnesis OLD CARTS, dokumentasi visual, dan skrining lesi oral berbasis AI.</p>
                
                <div style="background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 16px; padding: 16px; margin-top: 30px; display: flex; gap: 15px;">
                    <span style="font-size: 1.2rem;">🛡️</span>
                    <p style="margin: 0; font-size: 0.9rem; line-height: 1.5;">Dirancang sebagai pendukung keputusan klinis. Setiap temuan perlu diverifikasi oleh dokter gigi.</p>
                </div>
            </div>
        </div>
        <div class="login-bg-right"></div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_right:
        st.markdown("<div style='height: 18vh;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="fade-up" style="padding: 0 15%; z-index: 1; position: relative;">
                <p style="color: {C_BLUE}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08rem; margin-bottom: 5px;">SELAMAT DATANG</p>
                <h2 style="color: {C_INK} !important; font-weight: 800; font-size: 1.8rem; margin-top: 0;">Masuk ke ruang klinis Anda</h2>
                <p style="color: {C_MUTED}; font-size: 0.95rem; margin-bottom: 30px;">Gunakan identitas operator untuk mengakses dashboard skrining.</p>
            </div>
        """, unsafe_allow_html=True)
        
        _, col_form, _ = st.columns([1.5, 7, 1.5])
        with col_form:
            with st.form("login_form"):
                st.markdown(f"<label style='font-weight: 600; color: {C_INK}; font-size: 0.9rem;'>Email / ID Operator</label>", unsafe_allow_html=True)
                user_input = st.text_input("ID", label_visibility="collapsed", placeholder="contoh: adinara savero")
                
                st.markdown(f"<br><label style='font-weight: 600; color: {C_INK}; font-size: 0.9rem;'>Password</label>", unsafe_allow_html=True)
                pass_input = st.text_input("PASS", label_visibility="collapsed", type="password", placeholder="••••••••")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit_btn = st.form_submit_button("Login Now")
                
                if submit_btn:
                    if user_input.lower().strip() == OPERATOR_ID and pass_input == OPERATOR_PASS:
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("Kredensial tidak valid. Periksa kembali ID operator dan password Anda.")
            
            st.markdown(f"""
                <div class="fade-up" style="margin-top: 25px; background: #faf8ff; border: 1px solid #e8defe; border-radius: 12px; padding: 16px;">
                    <p style="margin: 0; font-weight: 700; color: #6d43ca; font-size: 0.85rem;">Akses Demo</p>
                    <p style="margin: 5px 0 0 0; color: {C_MUTED}; font-size: 0.85rem;">ID: adinara savero · Password: 2560171013</p>
                </div>
            """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# CSS UTAMA & SIDEBAR (FROST UI SETELAH LOGIN)
# ============================================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Work Sans', sans-serif !important; color: {C_INK}; }}
    .stApp {{ background-color: {C_PAPER}; }}
    
    @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: none; }} }}
    .fade-up {{ animation: fadeUp 0.55s ease both; }}
    
    h1, h2, h3, h4, h5 {{ color: {C_INK} !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
    p, label {{ color: {C_MUTED}; }}
    
    /* FROST SIDEBAR */
    section[data-testid="stSidebar"] {{ background-color: {C_SIDE} !important; border-right: none !important; padding-top: 10px; }}
    section[data-testid="stSidebar"] * {{ color: #cbd5e1 !important; }} 
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
    
    /* NAVIGATION MENU STYLING */
    div.row-widget.stRadio > div {{ gap: 5px; }}
    div.row-widget.stRadio > div > label {{
        background-color: transparent; padding: 12px 16px; border-radius: 12px; transition: 0.2s ease; cursor: pointer;
    }}
    div.row-widget.stRadio > div > label:hover {{ background-color: rgba(161,99,247,0.17); }}
    div.row-widget.stRadio > div > label p {{ font-weight: 500 !important; font-size: 0.9rem !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{ 
        background-color: rgba(161,99,247,0.17) !important;
        box-shadow: inset 4px 0 0 {C_CYAN};
    }}
    div.row-widget.stRadio > div > label[data-checked="true"] p {{ color: #FFFFFF !important; font-weight: 600 !important; }}
    
    /* CLINICAL CARD (GLASS/FROST) */
    .clinical-card {{
        background: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 18px; padding: 24px;
        box-shadow: 0 8px 26px rgba(30,41,59,0.055); margin-bottom: 24px;
    }}
    
    /* BUTTONS */
    .stButton>button {{
        background: linear-gradient(135deg, #8768f5, #5d83fa) !important; color: white !important;
        border-radius: 12px !important; border: none !important; padding: 12px 24px !important;
        font-weight: 600 !important; transition: 0.2s !important; box-shadow: 0 8px 18px rgba(111,136,252,0.24) !important;
    }}
    .stButton>button:hover {{ transform: translateY(-2px); filter: brightness(1.04); }}
    
    /* BADGES */
    .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; display: inline-block; }}
    .badge-low {{ background-color: #d1fae5; color: #047857; }}
    .badge-med {{ background-color: #fef3c7; color: #b45309; }}
    .badge-high {{ background-color: #fee2e2; color: #b91c1c; }}
    .badge-conf-high {{ background-color: #d1fae5; color: #047857; border: 1px solid #10b981; }}
    .badge-conf-med {{ background-color: #fef3c7; color: #b45309; border: 1px solid #f59e0b; }}
    .badge-conf-low {{ background-color: #fee2e2; color: #b91c1c; border: 1px solid #ef4444; }}
    
    /* REMOVE HEADER/FOOTER */
    #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV & SETTINGS ---
with st.sidebar:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 30px; padding: 0 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 25px;">
            <div style="width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(to bottom right, {C_VIOLET}, {C_CYAN}); display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; color: white;">✦</div>
            <div>
                <h3 style="margin:0; font-size: 1.2rem; color: white !important; font-weight: 700;">{CLINIC_NAME}</h3>
                <p style="margin:0; font-size: 0.75rem; color: #94a3b8 !important;">AI Medical Dashboard</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", ["Dashboard Skrining", "Rekam Medis (EMR)", "Analitik Kinerja", "Referensi Klinis"], label_visibility="collapsed")
    
    st.markdown("<div style='border-top: 1px solid rgba(255,255,255,0.1); margin-top: 25px; padding-top: 20px;'><p style='font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08rem; color: #94a3b8 !important; text-transform: uppercase;'>Pengaturan Model</p></div>", unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.55, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: auto; padding-top: 25px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 12px;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {C_BLUE}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">AS</div>
            <div>
                <p style="margin: 0; color: white !important; font-weight: 600; font-size: 0.9rem;">Adinara Savero</p>
                <p style="margin: 0; color: #94a3b8 !important; font-size: 0.75rem;">{USER_ROLE}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Log out", key="logout_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.anamnesis_data = None
        st.rerun()

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN 1: DASHBOARD SKRINING
# ============================================================
if menu == "Dashboard Skrining":
    st.markdown("<div class='fade-up'>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: {C_BLUE}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08rem; margin-bottom: 5px; text-transform: uppercase;'>Dashboard Skrining</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top: 0; font-size: 2rem;'>Skrining Lesi Oral</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1rem; color: {C_MUTED}; margin-bottom: 25px;'>Sistem menggabungkan keluhan subjektif OLD CARTS dan inferensi visual sebagai pendukung keputusan klinis.</p>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 16px; padding: 16px 20px; display: flex; gap: 15px; margin-bottom: 25px;">
            <span style="font-size: 1.2rem;">⚠️</span>
            <p style="margin: 0; color: #78350f; font-size: 0.9rem; font-weight: 500; line-height: 1.5;"><strong>Peringatan klinis:</strong> Hasil yang ditampilkan adalah skrining pendukung berbasis Demo AI, bukan diagnosis final. Keputusan klinis, pemeriksaan langsung, dan penatalaksanaan harus dilakukan oleh dokter gigi.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- ANAMNESIS (OLD CARTS) ---
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px;">
            <div style="width: 36px; height: 36px; background: #f0ecff; color: #855ae5; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">📋</div>
            <div>
                <h2 style="margin: 0; font-size: 1.3rem;">Anamnesis OLD CARTS</h2>
                <p style="margin: 0; font-size: 0.85rem; color: {C_MUTED};">Lengkapi parameter keluhan subjektif pasien.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("form_old_carts"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            o_val = st.text_input("Onset (O) - Kapan gejala muncul?", placeholder="Misal: 3 hari lalu, bertahap")
            l_val = st.text_input("Location (L) - Lokasi & Penjalaran?", placeholder="Misal: Mukosa bukal kanan")
            d_val = st.text_input("Duration (D) - Durasi & Sifat?", placeholder="Misal: Terus-menerus / Hilang timbul")
            c_val = st.text_input("Character (C) - Karakteristik?", placeholder="Misal: Nyeri tajam, tumpul, berdarah")
        with c2:
            a_val = st.text_input("Aggravating (A) - Faktor pemicu?", placeholder="Misal: Saat makan pedas")
            r_val = st.text_input("Relieving (R) - Faktor pereda?", placeholder="Misal: Minum air dingin")
            t_val = st.text_input("Timing (T) - Waktu keparahan?", placeholder="Misal: Malam hari")
            s_val = st.slider("Severity (S) - Skala Nyeri 0-10", 0, 10, 0)
        
        if st.form_submit_button("Simpan Parameter Anamnesis"):
            st.session_state.anamnesis_data = {
                "O_Onset": o_val or "Tidak diisi", "L_Location": l_val or "Tidak diisi", 
                "D_Duration": d_val or "Tidak diisi", "C_Character": c_val or "Tidak diisi",
                "A_Aggravating": a_val or "Tidak diisi", "R_Relieving": r_val or "Tidak diisi", 
                "T_Timing": t_val or "Tidak diisi", "S_Severity": s_val
            }
            st.success("Parameter anamnesis tersimpan sementara. Silakan lanjut ke Fase Akuisisi Visual.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- AKUISISI VISUAL BATCH ---
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px;">
            <div style="width: 36px; height: 36px; background: #e7faff; color: #1688a4; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">📸</div>
            <div>
                <h2 style="margin: 0; font-size: 1.3rem;">Fase Akuisisi Visual</h2>
                <p style="margin: 0; font-size: 0.85rem; color: {C_MUTED};">Unggah satu atau beberapa citra klinis untuk dianalisis.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Unggah Citra (Batch)", "Kamera Perangkat"])
    images_to_process, file_names = [], []

    with tabs[0]:
        uploaded_files = st.file_uploader("Format JPG/PNG (Bisa lebih dari satu)", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
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
        st.markdown("<hr style='border: none; border-top: 1px dashed #e8edf2; margin: 25px 0;'>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Jalankan Analisis AI & Sintesis Diagnosis", use_container_width=True, disabled=(model is None))
        
        if analyze_btn:
            if not st.session_state.anamnesis_data:
                st.warning("Peringatan: Anda belum menyimpan data Anamnesis OLD CARTS. Hasil sintesis hanya akan berdasar pada deteksi visual.")
                anamnesis = {}
            else:
                anamnesis = st.session_state.anamnesis_data

            progress_bar = st.progress(0, text="Menginisialisasi analisis Batch Demo AI...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress_bar.progress((idx + 1) / len(images_to_process), text=f"Memproses citra: {f_name}...")
                results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                
                res_plotted = results[0].plot()
                boxes = results[0].boxes
                detections = []
                
                for box in boxes:
                    c_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    n_lesi = model.names[c_id]
                    detections.append((n_lesi, conf))
                
                sintesis_akhir = synthesize_clinical_diagnosis(detections, anamnesis)
                
                record_base = {
                    "ID": f"EMR-{str(uuid.uuid4())[:6].upper()}",
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

                # KARTU HASIL (Mirip HTML Template)
                st.markdown(f"""
                    <div class="clinical-card" style="margin-top: 25px; padding: 25px;">
                        <h3 style="margin-top: 0; font-size: 1.2rem; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 15px; margin-bottom: 20px;">
                            Hasil Skrining: <span style="font-weight: 500; color: {C_MUTED};">{f_name}</span>
                        </h3>
                """, unsafe_allow_html=True)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown(f"<p style='font-size: 0.75rem; font-weight: 700; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 0.05rem; margin-bottom: 8px;'>Citra Asli</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_img2:
                    st.markdown(f"<p style='font-size: 0.75rem; font-weight: 700; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 0.05rem; margin-bottom: 8px;'>Overlay Deteksi Demo AI</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True)

                if detections:
                    st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
                    for n_lesi, conf in sorted(detections, key=lambda x: -x[1]):
                        info = get_lesion_info(n_lesi)
                        urg = info['urgensi']
                        urg_badge = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
                        
                        conf_pct = conf * 100
                        conf_badge = "badge-conf-high" if conf_pct >= 75 else "badge-conf-med" if conf_pct >= 50 else "badge-conf-low"

                        st.markdown(f"""
                            <div style="margin-bottom: 15px;">
                                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                                    <span class="badge {urg_badge}">Urgensi: {urg}</span>
                                    <h4 style="margin: 0; font-size: 1.1rem;">{info['nama_klinis']}</h4>
                                </div>
                                <span class="badge {conf_badge}" style="margin-bottom: 10px; font-size: 0.7rem;">Confidence Demo AI: {conf_pct:.1f}%</span>
                                <p style="font-size: 0.9rem; color: {C_MUTED}; line-height: 1.5; margin-bottom: 15px;">{info['deskripsi']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                    <div style="background: #f6f4ff; border-radius: 12px; padding: 16px; margin-top: 10px;">
                        <p style="font-size: 0.75rem; font-weight: 700; color: {C_VIOLET}; text-transform: uppercase; margin: 0 0 8px 0;">Sintesis Suspek Diagnosis (AI + OLD CARTS)</p>
                        <p style="margin: 0; font-size: 0.95rem; font-weight: 500; line-height: 1.5;">{sintesis_akhir}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            progress_bar.progress(1.0, text="Skrining selesai. Data tersimpan ke EMR.")
            append_log(all_new_records)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 2: REKAM MEDIS (EMR)
# ============================================================
elif menu == "Rekam Medis (EMR)":
    st.markdown("<div class='fade-up'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {C_BLUE}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08rem; margin-bottom: 5px; text-transform: uppercase;'>Rekam Medis</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top: 0; font-size: 2rem;'>Rekam Medis (EMR)</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1rem; color: {C_MUTED}; margin-bottom: 25px;'>Riwayat hasil skrining yang tersimpan dalam basis data sistem.</p>", unsafe_allow_html=True)
    
    df_log = load_log()
    if df_log.empty:
        st.info("Basis data rekam medis kosong.")
    else:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            opts = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Filter Lesi / Anomali", opts, default=opts)
        with col_f2:
            tgl_s = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            date_range = st.date_input("Filter Waktu Pemeriksaan", value=(tgl_s.min(), tgl_s.max())) if not tgl_s.empty else None
        with col_f3:
            min_conf = st.slider("Confidence Minimum (%)", 0, 100, 0)
        st.markdown("</div>", unsafe_allow_html=True)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= (min_conf / 100.0))
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)

        df_filtered = df_log[mask]
        
        st.dataframe(df_filtered, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh CSV", data=csv, file_name="EMR_RSGM_Unjani.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("📥 Unduh Excel", data=buffer.getvalue(), file_name="EMR_RSGM_Unjani.xlsx", use_container_width=True)
            except ImportError: pass
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 3: ANALITIK KINERJA
# ============================================================
elif menu == "Analitik Kinerja":
    st.markdown("<div class='fade-up'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {C_BLUE}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08rem; margin-bottom: 5px; text-transform: uppercase;'>Analitik Kinerja</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top: 0; font-size: 2rem;'>Analitik Kinerja</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1rem; color: {C_MUTED}; margin-bottom: 25px;'>Ringkasan pola temuan dan distribusi dari rekam skrining pada sistem.</p>", unsafe_allow_html=True)
    
    df_log = load_log()
    
    if df_log.empty:
        st.info("Memerlukan minimal 1 data entri untuk menghasilkan grafis analitik.")
    else:
        tot = len(df_log)
        avg = df_log["Confidence"].mean() * 100
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(f"""
            <div class='clinical-card'>
                <p style='margin: 0; font-size: 0.9rem; font-weight: 600; color: {C_MUTED}; text-transform: uppercase;'>Total Kasus Tercatat</p>
                <p style='margin: 10px 0 0 0; font-size: 2.5rem; font-weight: 800; color: {C_INK};'>{tot}</p>
            </div>""", unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
            <div class='clinical-card'>
                <p style='margin: 0; font-size: 0.9rem; font-weight: 600; color: {C_MUTED}; text-transform: uppercase;'>Rata-rata Confidence AI</p>
                <p style='margin: 10px 0 0 0; font-size: 2.5rem; font-weight: 800; color: {C_BLUE};'>{avg:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("<div class='clinical-card'><h3 style='margin-top:0;'>📊 Prevalensi Lesi Positif</h3>", unsafe_allow_html=True)
            df_valid = df_log[df_log["Lesi_Terdeteksi"] != "Tidak terdeteksi"]
            if not df_valid.empty:
                st.bar_chart(df_valid["Lesi_Terdeteksi"].value_counts(), color=C_BLUE)
            else:
                st.info("Belum ada data lesi positif yang terekam.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_c2:
            st.markdown("<div class='clinical-card'><h3 style='margin-top:0;'>🔥 Distribusi Severity Nyeri</h3>", unsafe_allow_html=True)
            if "S_Severity" in df_log.columns and not df_log["S_Severity"].dropna().empty:
                st.bar_chart(df_log["S_Severity"].value_counts().sort_index(), color=C_CYAN)
            else:
                st.info("Data tingkat nyeri belum tersedia.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='clinical-card'><h3 style='margin-top:0;'>📈 Tren Skrining Harian</h3>", unsafe_allow_html=True)
        st.line_chart(df_log.groupby("Tanggal").size(), color=C_VIOLET)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 4: REFERENSI ENSIKLOPEDIA
# ============================================================
elif menu == "Referensi Klinis":
    st.markdown("<div class='fade-up'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {C_BLUE}; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08rem; margin-bottom: 5px; text-transform: uppercase;'>Ensiklopedia Edukatif</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top: 0; font-size: 2rem;'>Referensi Klinis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1rem; color: {C_MUTED}; margin-bottom: 25px;'>Referensi ringkas untuk membantu orientasi temuan lesi oral pada 8 kelas deteksi proyek ini.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 16px; padding: 16px 20px; margin-bottom: 25px;">
            <p style="margin: 0; color: #0c4a6e; font-size: 0.9rem; font-weight: 500;">Konten referensi ini bersifat edukatif dan perlu diverifikasi dengan pedoman klinis terkini serta penilaian dokter gigi.</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    idx = 0
    for key, info in LESION_INFO.items():
        urg = info['urgensi']
        urg_badge = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
        
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="clinical-card" style="border-top: 4px solid {C_VIOLET}; padding: 20px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <h3 style="margin: 0; font-size: 1.1rem;">{info['nama_klinis']}</h3>
                        <span class="badge {urg_badge}" style="font-size: 0.65rem;">{urg}</span>
                    </div>
                    <p style="font-size: 0.85rem; color: {C_MUTED}; line-height: 1.5; margin-bottom: 15px;">{info['deskripsi']}</p>
                    <div style="background: #f7f8ff; padding: 12px; border-radius: 10px;">
                        <p style="margin: 0 0 5px 0; font-size: 0.7rem; font-weight: 700; color: {C_BLUE}; text-transform: uppercase;">Rekomendasi</p>
                        <p style="margin: 0; font-size: 0.85rem; color: #475569;">{info['rekomendasi']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        idx += 1
    st.markdown("</div>", unsafe_allow_html=True)
