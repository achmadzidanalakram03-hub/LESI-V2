"""
Klinik AI RSGM — Sistem Skrining Lesi Oral & Anamnesis Terpadu
=============================================================
Antarmuka: Premium Frost UI (Glassmorphism), Animasi Smooth, Modern SaaS
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
# KONFIGURASI GLOBAL & STRUKTUR DATABASE
# ============================================================
APP_VERSION = "3.5 (Enterprise EMR & Frost UI)"
CLINIC_NAME = "RSGM Unjani"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi_emr.csv")

# Database diperluas untuk mencakup data OLD CARTS
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi", "Confidence", "Nama_File",
    "O_Onset", "L_Location", "D_Duration", "C_Character", 
    "A_Aggravating", "R_Relieving", "T_Timing", "S_Severity", "Suspek_Diagnosis"
]

# PALET WARNA MODERN MEDICAL
PRIMARY = "#0d9488"       # Teal 600
PRIMARY_LIGHT = "#f0fdfa" # Teal 50
ACCENT = "#0ea5e9"        # Sky 500
DARK_TEXT = "#0f172a"     # Slate 900
GRAY_TEXT = "#64748b"     # Slate 500
BG_GRADIENT = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"

LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum",
        "deskripsi": "Lesi traumatik akibat gigitan berulang pada mukosa pipi.",
        "rekomendasi": "Edukasi hilangkan habit; evaluasi 2 minggu.",
        "urgensi": "Rendah"
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue",
        "deskripsi": "Penumpukan debris keratin pada dorsum lidah.",
        "rekomendasi": "Instruksi pembersihan dengan tongue scraper.",
        "urgensi": "Rendah"
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Demineralisasi jaringan keras gigi akibat bakteri.",
        "rekomendasi": "Rujuk untuk preparasi dan restorasi/perawatan saluran akar.",
        "urgensi": "Sedang-Tinggi"
    },
    "linea alba": {
        "nama_klinis": "Linea Alba Buccalis",
        "deskripsi": "Garis hiperkeratosis putih sejajar bidang oklusal.",
        "rekomendasi": "Lesi jinak, tidak perlu intervensi khusus.",
        "urgensi": "Rendah"
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran pembuluh darah vena di ventral lidah.",
        "rekomendasi": "Observasi klinis, fisiologis pada usia lanjut.",
        "urgensi": "Rendah"
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Deposit mineral plak terkalsifikasi.",
        "rekomendasi": "Tindakan scaling dan root planing (SRP).",
        "urgensi": "Sedang"
    },
    "torus": {
        "nama_klinis": "Torus",
        "deskripsi": "Eksostosis tulang jinak asimtomatik.",
        "rekomendasi": "Observasi kecuali mengganggu pembuatan protesa.",
        "urgensi": "Rendah"
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Hilangnya lapisan epitel akibat trauma.",
        "rekomendasi": "Kendalikan faktor etiologi, berikan obat topikal.",
        "urgensi": "Sedang"
    },
}

# --- INISIALISASI SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'anamnesis_data' not in st.session_state:
    st.session_state.anamnesis_data = None

# --- KONFIGURASI HALAMAN & INJEKSI CSS MODERN (ANIMASI + FROST UI) ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
    /* Font Startup Modern: Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
    
    .stApp {{ background: {BG_GRADIENT}; }}
    
    /* ANIMASI KUSTOM */
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    
    .anim-fade {{ animation: fadeIn 0.8s ease-out forwards; }}
    .anim-slide {{ animation: slideUp 0.6s ease-out forwards; }}
    
    /* TIPOGRAFI */
    h1, h2, h3, h4, h5, h6 {{ color: {DARK_TEXT} !important; font-weight: 700 !important; letter-spacing: -0.03em; }}
    p, label {{ color: {GRAY_TEXT}; }}
    
    /* FROST UI / GLASSMORPHISM UNTUK SIDEBAR & KARTU */
    [data-testid="stSidebar"] {{ 
        background: rgba(255, 255, 255, 0.6) !important; 
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
        transition: transform 0.3s ease;
    }}
    .glass-card:hover {{ transform: translateY(-3px); }}
    
    /* METRIK */
    .metric-label {{ font-size: 0.85rem; color: {GRAY_TEXT}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-value {{ font-size: 2.5rem; font-weight: 700; color: {PRIMARY}; line-height: 1.1; margin: 8px 0; }}
    
    /* TOMBOL */
    .stButton>button {{
        background: {PRIMARY} !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 14px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.25) !important;
    }}
    .stButton>button:hover {{
        background: #0f766e !important; 
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.4) !important; 
        transform: translateY(-2px) !important;
    }}
    
    /* BADGES */
    .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; display: inline-block; }}
    .badge-low {{ background-color: #d1fae5; color: #065f46; }}
    .badge-med {{ background-color: #fef3c7; color: #92400e; }}
    .badge-high {{ background-color: #fee2e2; color: #991b1b; }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HALAMAN LOGIN (SISTEM AUTENTIKASI)
# ============================================================
if not st.session_state.logged_in:
    st.markdown("""
        <div style="height: 15vh;"></div>
        <div class="anim-slide" style="max-width: 400px; margin: auto; background: rgba(255,255,255,0.85); backdrop-filter: blur(20px); border-radius: 24px; padding: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.08); border: 1px solid rgba(255,255,255,0.5); text-align: center;">
            <h2 style="color: #0d9488 !important; margin-bottom: 5px;">Portal Klinis AI</h2>
            <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 30px;">RSGM Universitas Jenderal Achmad Yani</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            user_input = st.text_input("Identitas Operator", placeholder="Masukkan nama pengguna...")
            pass_input = st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi akses...")
            submit_btn = st.form_submit_button("Masuk Sistem", use_container_width=True)
            
            if submit_btn:
                if user_input.lower().strip() == "adinara savero" and pass_input == "2560171013":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Kredensial tidak valid. Akses ditolak.")
    st.stop()


# ============================================================
# FUNGSI BANTU & LOGIKA AI (SINTESIS KLINIS)
# ============================================================
def init_db() -> None:
    if not DB_FILE.exists():
        pd.DataFrame(columns=DB_COLUMNS).to_csv(DB_FILE, index=False)
        return
    try:
        df = pd.read_csv(DB_FILE)
    except Exception:
        df = pd.DataFrame(columns=DB_COLUMNS)
    
    # Penyesuaian skema jika ada file lama
    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing or list(df.columns) != DB_COLUMNS:
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

def synthesize_clinical_diagnosis(detections, anamnesis):
    """
    Logika Core: Mensintesis data visual (AI) dan keluhan pasien (OLD CARTS)
    """
    if not detections:
        return "Tidak teridentifikasi kelainan struktural secara visual. Lakukan pendekatan simptomatik murni berdasarkan anamnesis."
    
    hasil_sintesis = []
    
    for nama_lesi, _ in detections:
        lesi_key = nama_lesi.lower()
        if anamnesis:
            sev = anamnesis.get("S_Severity", 0)
            char = str(anamnesis.get("C_Character", "")).lower()
            timing = str(anamnesis.get("T_Timing", "")).lower()
            
            if lesi_key == "karies":
                if sev >= 7 or "denyut" in char or "malam" in timing:
                    hasil_sintesis.append("Suspek Pulpitis Irreversibel (Karies profunda terdeteksi + keluhan nyeri spontan/berdenyut parah).")
                elif sev >= 4 or "ngilu" in char:
                    hasil_sintesis.append("Suspek Pulpitis Reversibel (Karies terdeteksi + ngilu terpicu stimulus).")
                else:
                    hasil_sintesis.append("Karies Gigi asimtomatik (Terkonfirmasi visual tanpa gejala subjektif signifikan).")
            
            elif lesi_key in ["ulkus traumatikus", "cheek biting"]:
                if sev >= 5:
                    hasil_sintesis.append(f"Ulserasi/Stomatitis Traumatik akut (Nyeri skala {sev}/10). Berikan kortikosteroid topikal & hilangkan f. etiologi.")
                else:
                    hasil_sintesis.append(f"Lesi traumatik ({LESION_INFO[lesi_key]['nama_klinis']}) tahap penyembuhan / ringan.")
            
            elif lesi_key == "stain calculus":
                if "berdarah" in char:
                    hasil_sintesis.append("Suspek Periodontitis / Gingivitis Marginal (Deposit kalkulus + indikasi inflamasi jaringan lunak).")
                else:
                    hasil_sintesis.append("Kalkulus tanpa komplikasi peradangan akut yang dilaporkan pasien.")
            else:
                hasil_sintesis.append(f"Terkonfirmasi visual: {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            hasil_sintesis.append(f"Deteksi visual murni (Tanpa anamnesis): {LESION_INFO.get(lesi_key, {}).get('nama_klinis', nama_lesi)}.")
            
    return " | ".join(hasil_sintesis)


# ============================================================
# LAYOUT UTAMA: SIDEBAR (FROST UI)
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="margin-bottom: 40px; margin-top: 20px;">
            <div style="width: 40px; height: 40px; background: {PRIMARY}; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                <span style="color: white; font-weight: bold; font-size: 20px;">+</span>
            </div>
            <h2 style="margin:0; font-size: 1.2rem; color: {PRIMARY} !important;">{CLINIC_NAME}</h2>
            <p style="margin:0; font-size: 0.85rem; font-weight: 500;">AI Vision & EMR System</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi Sistem", ["Skrining & Anamnesis", "Rekam Medis (EMR)", "Analitik Kinerja", "Referensi Klinis"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("**Parameter Neural Network**")
    conf_threshold = st.slider("Batas Kepercayaan (Conf)", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("Batas Tumpang Tindih (IoU)", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 40px; padding: 16px; background: rgba(255,255,255,0.7); border-radius: 12px; border: 1px solid rgba(0,0,0,0.05);">
            <p style="margin: 0; font-size: 0.7rem; color: {GRAY_TEXT}; font-weight: 700; text-transform: uppercase;">Operator Sistem</p>
            <p style="margin: 4px 0 0 0; font-weight: 700; color: {DARK_TEXT}; font-size: 0.9rem;">adinara savero</p>
            <p style="margin: 0; font-size: 0.75rem; color: {PRIMARY}; font-weight: 600;">{USER_ROLE}</p>
            <div style="margin-top: 15px;">
    """, unsafe_allow_html=True)
    
    if st.button("Keluar Sesi", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN 1: DASHBOARD (ANAMNESIS & DETEKSI AI)
# ============================================================
if menu == "Skrining & Anamnesis":
    st.markdown("<div class='anim-slide'>", unsafe_allow_html=True)
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(f"<h1 style='margin-bottom: 4px;'>Dashboard Diagnosis Terpadu</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; margin-top: 0;'>Sintesis keluhan pasien (OLD CARTS) dan inferensi visual berbasis YOLO.</p>", unsafe_allow_html=True)
    with col_hdr2:
        st.markdown(f"<div style='text-align: right; margin-top: 16px;'><span style='background: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; color: {PRIMARY}; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>{datetime.now().strftime('%d %b %Y')}</span></div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: {PRIMARY_LIGHT}; border-left: 5px solid {PRIMARY}; padding: 16px 20px; border-radius: 12px; margin: 24px 0; box-shadow: 0 4px 6px rgba(13,148,136,0.05);">
            <p style="margin: 0; color: {PRIMARY}; font-size: 0.9rem; font-weight: 500;">
                <strong>DISCLAIMER KLINIS:</strong> Sistem kecerdasan buatan ini berfungsi secara eksklusif sebagai alat bantu skrining dan dokumentasi. Diagnosis definitif dan penetapan rencana perawatan tetap menjadi otoritas dokter gigi yang menangani.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- BAGIAN 1: ANAMNESIS (OLD CARTS) ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Fase 1: Wawancara Medis (OLD CARTS)</h3>", unsafe_allow_html=True)
    
    with st.form("form_old_carts"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            o_val = st.text_input("Onset (O)", placeholder="Kapan gejala mulai muncul? Mendadak/bertahap?")
            l_val = st.text_input("Location (L)", placeholder="Di sebelah mana? Apakah menjalar?")
            d_val = st.text_input("Duration (D)", placeholder="Berapa lama gejala bertahan?")
            c_val = st.selectbox("Character (C)", ["", "Berdenyut tajam", "Ngilu/Tumpul", "Terbakar/Panas", "Gatal", "Berdarah", "Lainnya"])
        with c2:
            a_val = st.text_input("Aggravating (A)", placeholder="Apa yang memperburuk? (Misal: makan manis/dingin)")
            r_val = st.text_input("Relieving (R)", placeholder="Apa yang meredakan? (Misal: minum obat/istirahat)")
            t_val = st.selectbox("Timing (T)", ["", "Sepanjang waktu", "Malam hari saja", "Pagi hari", "Saat/setelah makan"])
            s_val = st.slider("Severity (S) - Visual Analog Scale", 0, 10, 0, help="0: Tidak nyeri, 10: Nyeri terhebat yang tak tertahankan")
        
        if st.form_submit_button("Rekam Parameter Anamnesis", use_container_width=True):
            st.session_state.anamnesis_data = {
                "O_Onset": o_val, "L_Location": l_val, "D_Duration": d_val, "C_Character": c_val,
                "A_Aggravating": a_val, "R_Relieving": r_val, "T_Timing": t_val, "S_Severity": s_val
            }
            st.success("Parameter berhasil direkam ke dalam memori sesi. Lanjutkan ke fase pemindaian visual.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- BAGIAN 2: DETEKSI VISUAL ---
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>Fase 2: Pemindaian Lesi Intraoral</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Unggah Citra Medis (Batch)", "Akuisisi Langsung (Kamera)"])
    images_to_process, file_names = [], []

    with tabs[0]:
        uploaded_files = st.file_uploader("Seret berkas JPG/PNG ke sini (Dapat memilih lebih dari 1 file)", accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    images_to_process.append(Image.open(f).convert("RGB"))
                    file_names.append(f.name)
                except Exception: pass

    with tabs[1]:
        camera_file = st.camera_input("Fokuskan lesi pada *viewfinder*")
        if camera_file is not None:
            try:
                images_to_process.append(Image.open(camera_file).convert("RGB"))
                file_names.append(f"Cam_{datetime.now().strftime('%H%M%S')}.jpg")
            except Exception: pass

    if images_to_process:
        st.markdown("---")
        if st.button("🚀 Jalankan Inferensi AI & Sinkronisasi Suspek", use_container_width=True, disabled=(model is None)):
            progress_bar = st.progress(0, text="Menginisialisasi bobot saraf...")
            all_new_records = []
            
            for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                progress_bar.progress((idx + 1) / len(images_to_process), text=f"Menganalisis matriks {f_name}...")
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
                
                # SINTESIS
                sintesis_akhir = synthesize_clinical_diagnosis(detections, anamnesis)
                
                # Rekam data
                if not detections:
                    # Jika tidak ada deteksi tapi anamnesis ada, tetap catat EMR
                    all_new_records.append({
                        "ID": str(uuid.uuid4())[:8], "Waktu": datetime.now().strftime("%H:%M:%S"),
                        "Tanggal": datetime.now().strftime("%Y-%m-%d"), "Lesi_Terdeteksi": "Tidak terdeteksi",
                        "Confidence": 0, "Nama_File": f_name, **anamnesis, "Suspek_Diagnosis": sintesis_akhir
                    })
                else:
                    for n_lesi, conf in detections:
                        all_new_records.append({
                            "ID": str(uuid.uuid4())[:8], "Waktu": datetime.now().strftime("%H:%M:%S"),
                            "Tanggal": datetime.now().strftime("%Y-%m-%d"), "Lesi_Terdeteksi": n_lesi,
                            "Confidence": round(conf, 4), "Nama_File": f_name, **anamnesis, "Suspek_Diagnosis": sintesis_akhir
                        })

                # RENDER HASIL VISUAL
                st.markdown(f"<h4 style='margin-top: 30px; padding-top: 20px; border-top: 1px solid {BORDER_COLOR};'>Dokumen EMR: {f_name}</h4>", unsafe_allow_html=True)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;'>Foto Klinis Pasien</p>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True, output_format="JPEG")
                with col_img2:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;'>Pemetaan Obyek Visual (YOLO)</p>", unsafe_allow_html=True)
                    st.image(res_plotted, use_container_width=True, output_format="JPEG")

                # KARTU SINTESIS
                st.markdown(f"""
                    <div class="anim-slide" style="background: white; border: 1px solid {BORDER_COLOR}; border-left: 6px solid {PRIMARY}; border-radius: 12px; padding: 24px; margin-top: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.04);">
                        <p style="font-size: 0.8rem; text-transform: uppercase; color: {GRAY_TEXT}; font-weight: 700; margin: 0 0 10px 0; letter-spacing: 0.05em;">Sintesis Suspek Diagnosis (AI + OLD CARTS)</p>
                        <p style="font-size: 1.1rem; color: {DARK_TEXT}; font-weight: 600; margin: 0; line-height: 1.5;">{sintesis_akhir}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # KARTU RINCIAN LESI
                if detections:
                    st.markdown("<h5 style='margin-top: 20px;'>Profil Klinis Parameter Visual</h5>", unsafe_allow_html=True)
                    for n_lesi, conf in sorted(detections, key=lambda x: -x[1]):
                        info = get_lesion_info(n_lesi)
                        urg = info['urgensi']
                        bg_badge = "badge-high" if "Tinggi" in urg else "badge-med" if urg == "Sedang" else "badge-low"
                        
                        st.markdown(f"""
                            <div style="background: {BG_GRADIENT}; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 1px solid {BORDER_COLOR};">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <span style="font-weight: 700; font-size: 1.05rem; color: {DARK_TEXT};">{info['nama_klinis']}</span>
                                    <div>
                                        <span class="badge" style="background: white; color: {DARK_TEXT}; border: 1px solid {BORDER_COLOR}; margin-right: 8px;">Akurasi: {conf*100:.1f}%</span>
                                        <span class="badge {bg_badge}">{urg}</span>
                                    </div>
                                </div>
                                <p style="font-size: 0.9rem; color: {GRAY_TEXT}; margin-bottom: 12px;">{info['deskripsi']}</p>
                                <div style="background: rgba(255,255,255,0.7); border-radius: 8px; padding: 12px; font-size: 0.85rem; font-weight: 500; color: {PRIMARY};">
                                    Rencana Perawatan: {info['rekomendasi']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            progress_bar.progress(1.0, text="Seluruh matriks data berhasil direkam.")
            append_log(all_new_records)
            st.session_state.anamnesis_data = None # Reset formulir setelah direkam
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 2: REKAM MEDIS & EXPORT
# ============================================================
elif menu == "Rekam Medis (EMR)":
    st.markdown("<div class='anim-fade'>", unsafe_allow_html=True)
    st.markdown("## Electronic Medical Record (EMR)")
    st.markdown("Arsip sistematis yang memuat riwayat wawancara klinis (OLD CARTS) dan parameter visual radiografis/fotografis pasien.")
    
    df_log = load_log()
    if df_log.empty:
        st.info("Basis data rekam medis saat ini kosong.")
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

        # Proses Filter
        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= min_conf)
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            t_start, t_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (pd.to_datetime(df_log["Tanggal"], errors="coerce") >= t_start) & (pd.to_datetime(df_log["Tanggal"], errors="coerce") <= t_end)

        df_filtered = df_log[mask]
        
        # Tampilkan tabel elegan
        st.dataframe(df_filtered.style.background_gradient(cmap='Teal', subset=['Confidence']), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Dokumen CSV", data=csv, file_name="EMR_RSGM_Unjani.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_filtered.to_excel(writer, index=False)
                st.download_button("Unduh Dokumen Excel", data=buffer.getvalue(), file_name="EMR_RSGM_Unjani.xlsx", use_container_width=True)
            except ImportError: pass
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 3: ANALITIK KINERJA
# ============================================================
elif menu == "Analitik Kinerja":
    st.markdown("<div class='anim-fade'>", unsafe_allow_html=True)
    st.markdown("## Analitik & Prevalensi Klinis")
    df_log = load_log()
    
    if df_log.empty:
        st.info("Memerlukan minimal 1 data entri untuk menghasilkan grafis analitik.")
    else:
        tot = len(df_log)
        avg = df_log["Confidence"].mean() * 100
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-label'>Volume Pemeriksaan</div>
                <div class='metric-value'>{tot}</div>
                <div style='color: {GRAY_TEXT}; font-size: 0.8rem;'>Tercatat di Sistem</div>
            </div>""", unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-label'>Rerata Presisi AI</div>
                <div class='metric-value'>{avg:.1f}%</div>
                <div style='color: {GRAY_TEXT}; font-size: 0.8rem;'>Tingkat Confidence</div>
            </div>""", unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Prevalensi Temuan Klinis</h4>", unsafe_allow_html=True)
            st.bar_chart(df_log["Lesi_Terdeteksi"].value_counts(), color=PRIMARY)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_c2:
            st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Tren Deteksi Kumulatif</h4>", unsafe_allow_html=True)
            st.line_chart(df_log.groupby("Tanggal").size(), color=ACCENT)
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
            <div class="glass-card" style="padding: 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 16px; margin-bottom: 16px;">
                    <h3 style="margin: 0; color: {PRIMARY} !important;">{info['nama_klinis']}</h3>
                    <span class="badge {bg_badge}">{urg}</span>
                </div>
                <p style="color: {DARK_TEXT}; font-size: 1rem; margin-bottom: 16px; font-weight: 500;">{info['deskripsi']}</p>
                <div style="background: {PRIMARY_LIGHT}; padding: 16px; border-radius: 10px; font-size: 0.9rem; color: #065f46; border-left: 4px solid {PRIMARY};">
                    <strong>Rekomendasi Tindakan:</strong> {info['rekomendasi']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
