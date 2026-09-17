"""
MAMMOUTH — MY ASSISTANT IN MOUTH HEALTH
=========================================================
Antarmuka: Frost UI Premium, Clean, dan Modern SaaS
Fitur: Multi-User Auth, SQLite Database, Data Isolation, OLD CARTS, Batch Upload, AI Synthesis
"""

import io
import uuid
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
APP_VERSION = "5.0 (Independent Multi-User System)"
APP_NAME = "MAMMOUTH"
DB_FILE = "mammouth.db"

# PALET WARNA (Modern Frost UI Medical SaaS)
PRIMARY = "#0f766e"        # Deep Teal
PRIMARY_LIGHT = "#ccfbf1"
ACCENT = "#fbbf24"         # Amber
DANGER = "#ef4444"         # Red
DARK_TEXT = "#0f172a"      # Slate 900
GRAY_TEXT = "#64748b"      # Slate 500
BG_MIST = "#f8fafc"        # Lighter Frost background
BORDER_COLOR = "#e2e8f0"

LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum (Cheek Biting)",
        "deskripsi": "Kebiasaan menggigit mukosa bukal, mengakibatkan tampilan kasar/bergerigi. Dapat berpotensi menyebabkan ulserasi. Sering terkait faktor stres psikologis.",
        "rekomendasi": "Edukasi penghentian kebiasaan buruk (habit breaking); evaluasi ulang bila lesi menetap >2 minggu.",
        "urgensi": "Rendah",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue",
        "deskripsi": "Permukaan lidah tertutup selaput pseudomembran akibat penumpukan debris, keratin tidak terdeskuamasi, dan mikroorganisme.",
        "rekomendasi": "Instruksikan pembersihan mekanis rutin (tongue scraper) dan evaluasi oral hygiene.",
        "urgensi": "Rendah",
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Demineralisasi jaringan keras gigi oleh asam hasil metabolisme bakteri plak.",
        "rekomendasi": "Pemeriksaan klinis (sondasi/perkusi) dan radiografis lanjutan untuk rencana restorasi atau perawatan saluran akar.",
        "urgensi": "Sedang-Tinggi",
    },
    "linea alba": {
        "nama_klinis": "Linea Alba Buccalis",
        "deskripsi": "Garis putih horizontal pada mukosa bukal setinggi bidang oklusal, umumnya akibat tekanan atau friksi oklusal ringan.",
        "rekomendasi": "Bersifat jinak dan fisiologis, umumnya tidak memerlukan tatalaksana khusus.",
        "urgensi": "Rendah",
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran vena (varises) pada permukaan ventral lidah, temuan umum pada individu usia lanjut.",
        "rekomendasi": "Tidak memerlukan tindakan invasif. Edukasi pasien terkait sifat jinak lesi.",
        "urgensi": "Rendah",
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Deposit terkalsifikasi (kalkulus) dan diskolorasi ekstrinsik pada permukaan gigi.",
        "rekomendasi": "Tindakan scaling dan root planing (SRP) profesional; instruksi DHE.",
        "urgensi": "Sedang",
    },
    "torus": {
        "nama_klinis": "Torus (Palatinus/Mandibularis)",
        "deskripsi": "Eksostosis tulang jinak, umumnya asimtomatik dan lambat membesar.",
        "rekomendasi": "Observasi. Pembedahan hanya diindikasikan bila mengganggu fungsi bicara/pengunyahan atau sebagai persiapan protesa.",
        "urgensi": "Rendah",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Lesi ulseratif mukosa oral sekunder akibat trauma mekanis (tergigit/gesekan), termal, atau kimiawi.",
        "rekomendasi": "Eliminasi faktor kausatif. Evaluasi ulang dalam 10-14 hari untuk menyingkirkan diagnosis banding keganasan.",
        "urgensi": "Sedang",
    },
}

# --- INISIALISASI SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'active_user' not in st.session_state:
    st.session_state.active_user = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'anamnesis_data' not in st.session_state:
    st.session_state.anamnesis_data = None
if 'yolo_version' not in st.session_state:
    st.session_state.yolo_version = "YOLOv8"

st.set_page_config(page_title=f"{APP_NAME} - AI Vision", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# FUNGSI DATABASE (SQLITE) & AUTENTIKASI
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Tabel Pengguna
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, 
                    password TEXT, 
                    full_name TEXT, 
                    role TEXT
                )''')
    # Tabel Log Pemeriksaan Terisolasi per Pengguna
    c.execute('''CREATE TABLE IF NOT EXISTS emr_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    waktu TEXT,
                    tanggal TEXT,
                    lesi_terdeteksi TEXT,
                    confidence REAL,
                    model_version TEXT,
                    nama_file TEXT,
                    o_onset TEXT,
                    l_location TEXT,
                    d_duration TEXT,
                    c_character TEXT,
                    a_aggravating TEXT,
                    r_relieving TEXT,
                    t_timing TEXT,
                    s_severity INTEGER,
                    suspek_diagnosis TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(username)
                )''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, full_name, role):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, full_name, role) VALUES (?, ?, ?, ?)", 
                  (username.lower(), hash_password(password), full_name, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_login(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, full_name, role FROM users WHERE username=? AND password=?", 
              (username.lower(), hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def load_user_logs(username) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM emr_logs WHERE user_id = ?"
    df = pd.read_sql(query, conn, params=(username,))
    conn.close()
    return df

def append_user_log(records: list):
    if not records: return
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for r in records:
        c.execute('''INSERT INTO emr_logs 
                     (log_id, user_id, waktu, tanggal, lesi_terdeteksi, confidence, model_version, nama_file, 
                      o_onset, l_location, d_duration, c_character, a_aggravating, r_relieving, t_timing, 
                      s_severity, suspek_diagnosis) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (r['ID'], r['user_id'], r['Waktu'], r['Tanggal'], r['Lesi_Terdeteksi'], r['Confidence'], 
                   r['Model_Version'], r['Nama_File'], r.get('O_Onset','-'), r.get('L_Location','-'), 
                   r.get('D_Duration','-'), r.get('C_Character','-'), r.get('A_Aggravating','-'), 
                   r.get('R_Relieving','-'), r.get('T_Timing','-'), r.get('S_Severity',0), r['Suspek_Diagnosis']))
    conn.commit()
    conn.close()

# Inisialisasi DB di awal
init_db()

# ============================================================
# FUNGSI AI & SINTESIS KLINIS
# ============================================================
@st.cache_resource(show_spinner="Memuat bobot model AI...")
def load_model(version: str):
    if YOLO is None: return None
    model_paths = {
        "YOLOv8": "best.pt",
        "YOLOv11": "yolov11_best.pt",
        "YOLOv12": "yolov12_best.pt"
    }
    path = Path(model_paths.get(version, "best.pt"))
    if not path.exists(): path = Path("best.pt")
    
    if path.exists():
        try: return YOLO(str(path))
        except: return None
    return None

def synthesize_clinical_diagnosis(detections, anamnesis):
    if not detections: return "Tidak terdeteksi anomali visual. Pertimbangkan observasi berbasis keluhan (OLD CARTS)."
    hasil = []
    for nama_lesi, _ in detections:
        key = nama_lesi.lower()
        if anamnesis:
            sev = int(anamnesis.get("S_Severity", 0))
            char = str(anamnesis.get("C_Character", "")).lower()
            
            if key == "karies":
                if sev >= 6 or "denyut" in char or "spontan" in char:
                    hasil.append("Suspek Pulpitis Irreversibel (Karies + Nyeri spontan/tajam).")
                elif sev >= 3 or "ngilu" in char or "manis" in char:
                    hasil.append("Suspek Pulpitis Reversibel (Karies + Hipersensitivitas).")
                else:
                    hasil.append("Karies Asimtomatik.")
            elif key in ["ulkus traumatikus", "cheek biting"]:
                if sev >= 5:
                    hasil.append(f"Lesi traumatik reaktif akut (VAS {sev}/10). Indikasi observasi ketat/topikal.")
                else:
                    hasil.append("Lesi traumatik fase penyembuhan / indolen.")
            else:
                hasil.append(f"{LESION_INFO.get(key, {}).get('nama_klinis', nama_lesi)}.")
        else:
            hasil.append(f"Deteksi Objek: {LESION_INFO.get(key, {}).get('nama_klinis', nama_lesi)}.")
    return " | ".join(hasil)

# ============================================================
# HALAMAN AUTENTIKASI (LOGIN & SIGN UP)
# ============================================================
if not st.session_state.logged_in:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
        [data-testid="stSidebar"] {{ display: none; }}
        .login-wrapper {{ min-height: 90vh; display: flex; align-items: center; justify-content: center; background-color: {BG_MIST};}}
        .login-card {{
            background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.6); border-radius: 24px;
            padding: 40px; width: 100%; max-width: 450px;
            box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.1);
        }}
        .stButton>button {{ background: {PRIMARY} !important; color: white !important; width: 100% !important; border-radius: 12px !important; padding: 12px !important; font-weight: 700 !important; margin-top: 10px; }}
        </style>
        <div class="login-wrapper">
            <div class="login-card">
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1 style="margin: 0; color: {PRIMARY}; font-weight: 800; font-size: 2.2rem;">{APP_NAME}</h1>
                    <p style="margin: 5px 0 0 0; color: {GRAY_TEXT}; font-weight: 600; font-size: 0.9rem;">Intelligent Oral Health Screening</p>
                </div>
    """, unsafe_allow_html=True)
    
    col_space1, col_form, col_space3 = st.columns([1, 6, 1])
    with col_form:
        tab_login, tab_register = st.tabs(["🔒 Masuk", "📝 Buat Akun"])
        
        with tab_login:
            with st.form("login_form"):
                log_user = st.text_input("Username", placeholder="Masukkan username Anda")
                log_pass = st.text_input("Password", type="password", placeholder="••••••••")
                if st.form_submit_button("MASUK SISTEM"):
                    user_data = verify_login(log_user, log_pass)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.active_user = user_data[0]
                        st.session_state.user_name = user_data[1]
                        st.session_state.user_role = user_data[2]
                        st.rerun()
                    else:
                        st.error("Username atau password salah.")
                        
        with tab_register:
            with st.form("register_form"):
                reg_name = st.text_input("Nama Lengkap", placeholder="Contoh: drg. John Doe")
                reg_role = st.text_input("Peran/Jabatan", placeholder="Contoh: Operator Klinis")
                reg_user = st.text_input("Username", placeholder="Pilih username unik")
                reg_pass = st.text_input("Password", type="password", placeholder="Buat password yang kuat")
                if st.form_submit_button("DAFTAR AKUN BARU"):
                    if reg_name and reg_role and reg_user and reg_pass:
                        success = create_user(reg_user, reg_pass, reg_name, reg_role)
                        if success:
                            st.success("Akun berhasil dibuat! Silakan kembali ke tab 'Masuk'.")
                        else:
                            st.error("Username sudah digunakan. Pilih username lain.")
                    else:
                        st.warning("Mohon lengkapi seluruh form pendaftaran.")
                        
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# CSS UTAMA (FROST UI SETELAH LOGIN)
# ============================================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
    .stApp {{ background-color: {BG_MIST}; }}
    
    .frost-card {{
        background: white; border: 1px solid {BORDER_COLOR}; border-radius: 16px;
        padding: 24px; margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .frost-card:hover {{ box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08); }}
    
    .badge {{ padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}
    .badge-high {{ background-color: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
    .badge-med {{ background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
    .badge-low {{ background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
    
    .stButton>button {{ background-color: {PRIMARY} !important; color: white !important; border-radius: 10px !important; font-weight: 600 !important; }}
    .stButton>button:hover {{ opacity: 0.9; }}
    
    [data-testid="stSidebar"] {{ background-color: white !important; border-right: 1px solid {BORDER_COLOR}; }}
    div.row-widget.stRadio > div > label {{ padding: 12px; border-radius: 8px; }}
    div.row-widget.stRadio > div > label[data-checked="true"] {{ background-color: {PRIMARY} !important; color: white !important; }}
    div.row-widget.stRadio > div > label[data-checked="true"] p {{ color: white !important; font-weight: 600 !important; }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h2 style="margin:0; font-size: 1.6rem; font-weight: 800; color:{PRIMARY};">{APP_NAME}</h2>
            <p style="margin:0; font-size: 0.75rem; font-weight: 600; color:{GRAY_TEXT};">Personal Workspace</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigasi", ["Dashboard Skrining", "Riwayat Klinis (EMR)", "Analitik Data", "Ensiklopedia Lesi", "Pengaturan Sistem"], label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #64748b;'>KONFIGURASI AI</p>", unsafe_allow_html=True)
    st.session_state.yolo_version = st.selectbox("Arsitektur Model", ["YOLOv8", "YOLOv11", "YOLOv12"], index=["YOLOv8", "YOLOv11", "YOLOv12"].index(st.session_state.yolo_version))
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU (NMS) Threshold", 0.05, 0.95, 0.45, 0.05)
    
    st.markdown(f"""
        <div style="margin-top: 30px; padding: 16px; background: {BG_MIST}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
            <p style="margin: 0; font-size: 0.7rem; color: {GRAY_TEXT}; font-weight: 700;">AKUN TERHUBUNG</p>
            <p style="margin: 4px 0 0 0; font-weight: 700; font-size: 0.9rem; color: {DARK_TEXT};">{st.session_state.user_name}</p>
            <p style="margin: 0; font-size: 0.75rem; color: {GRAY_TEXT};">{st.session_state.user_role}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Log Out", key="logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.active_user = None
        st.session_state.user_name = None
        st.rerun()

model = load_model(st.session_state.yolo_version)

# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard Skrining":
    st.markdown(f"<h2>Skrining Visual Otomatis</h2>", unsafe_allow_html=True)
    st.markdown("Penggabungan interpretasi Computer Vision dan Anamnesis untuk identifikasi anomali rongga mulut.")

    # --- ANAMNESIS OLD CARTS ---
    with st.expander("📝 Formulir Anamnesis (OLD CARTS)", expanded=(st.session_state.anamnesis_data is None)):
        with st.form("anamnesis_form"):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                o_val = st.text_input("Onset (O)", placeholder="Sejak kapan muncul?")
                l_val = st.text_input("Location (L)", placeholder="Lokasi anatomis lesi?")
                d_val = st.text_input("Duration (D)", placeholder="Durasi nyeri/keluhan?")
                c_val = st.text_input("Character (C)", placeholder="Sifat keluhan (tajam, tumpul, berdenyut)?")
            with c2:
                a_val = st.text_input("Aggravating (A)", placeholder="Faktor yang memperparah?")
                r_val = st.text_input("Relieving (R)", placeholder="Faktor yang meredakan?")
                t_val = st.text_input("Timing (T)", placeholder="Waktu munculnya keluhan?")
                s_val = st.slider("Severity (S) - Skala Nyeri (VAS) 0-10", 0, 10, 0)
            
            if st.form_submit_button("Simpan Anamnesis"):
                st.session_state.anamnesis_data = {
                    "O_Onset": o_val or "-", "L_Location": l_val or "-", "D_Duration": d_val or "-", 
                    "C_Character": c_val or "-", "A_Aggravating": a_val or "-", "R_Relieving": r_val or "-", 
                    "T_Timing": t_val or "-", "S_Severity": s_val
                }
                st.success("Anamnesis direkam. AI akan menggunakan konteks ini untuk sintesis diagnosis.")

    # --- AKUISISI VISUAL ---
    st.markdown("<div class='frost-card'>", unsafe_allow_html=True)
    st.markdown("<h4>📸 Akuisisi & Penyesuaian Citra Klinis</h4>", unsafe_allow_html=True)
    
    tab_unggah, tab_kamera = st.tabs(["Unggah Gambar", "Kamera Intraoral/Webcam"])
    raw_images, file_names = [], []

    with tab_unggah:
        uploaded_files = st.file_uploader("Pilih satu atau beberapa foto klinis", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    raw_images.append(Image.open(f).convert("RGB"))
                    file_names.append(f.name)
                except Exception: pass

    with tab_kamera:
        camera_file = st.camera_input("Ambil foto secara langsung")
        if camera_file:
            try:
                raw_images.append(Image.open(camera_file).convert("RGB"))
                file_names.append("Cam_" + datetime.now().strftime("%H%M%S") + ".jpg")
            except Exception: pass

    images_to_process = []
    if raw_images:
        st.markdown("---")
        st.markdown("**Penyesuaian Citra Digital** *(Gunakan GIMP untuk pra-pemrosesan presisi tinggi)*")
        col_adj1, col_adj2 = st.columns(2)
        with col_adj1: bright_factor = st.slider("Kecerahan (Brightness)", 0.5, 2.0, 1.0, 0.1)
        with col_adj2: cont_factor = st.slider("Kontras (Contrast)", 0.5, 2.0, 1.0, 0.1)

        for img in raw_images:
            enhancer = ImageEnhance.Brightness(img)
            img_adj = enhancer.enhance(bright_factor)
            enhancer = ImageEnhance.Contrast(img_adj)
            img_adj = enhancer.enhance(cont_factor)
            images_to_process.append(img_adj)

        analyze_btn = st.button(f"JALANKAN INFERENSI ({st.session_state.yolo_version})", use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            anamnesis = st.session_state.anamnesis_data or {}
            all_new_records = []
            
            with st.status("Memproses gambar klinis...", expanded=True) as status:
                for idx, (image, f_name) in enumerate(zip(images_to_process, file_names)):
                    st.write(f"Inferensi pada citra: {f_name}")
                    try:
                        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                        res_plotted = results[0].plot()
                        boxes = results[0].boxes
                        detections = [(model.names[int(b.cls[0].item())], float(b.conf[0].item())) for b in boxes]
                        
                        sintesis = synthesize_clinical_diagnosis(detections, anamnesis)
                        
                        # ID User dilekatkan pada record
                        base_record = {
                            "ID": str(uuid.uuid4())[:8].upper(),
                            "user_id": st.session_state.active_user,
                            "Waktu": datetime.now().strftime("%H:%M:%S"),
                            "Tanggal": datetime.now().strftime("%Y-%m-%d"),
                            "Model_Version": st.session_state.yolo_version,
                            "Nama_File": f_name,
                            **anamnesis,
                            "Suspek_Diagnosis": sintesis
                        }

                        if not detections:
                            all_new_records.append({**base_record, "Lesi_Terdeteksi": "Tidak terdeteksi", "Confidence": 0.0})
                        else:
                            for nama_lesi, conf_score in detections:
                                all_new_records.append({**base_record, "Lesi_Terdeteksi": nama_lesi, "Confidence": round(conf_score, 4)})

                        st.markdown(f"#### Hasil Analisis: {f_name}")
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.image(image, caption="Citra Input (Disesuaikan)", use_container_width=True)
                        with col_img2:
                            st.image(res_plotted, caption=f"Overlay Bounding Box ({st.session_state.yolo_version})", use_container_width=True)

                        if detections:
                            st.markdown(f"""
                            <div style="background: #eff6ff; padding: 16px; border-left: 4px solid #3b82f6; border-radius: 8px; margin-bottom: 20px;">
                                <strong>Sintesis Klinis (OLD CARTS + AI):</strong> {sintesis}
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gagal memproses {f_name}: {e}")
                
                append_user_log(all_new_records)
                status.update(label="Selesai! Data berhasil diekstrak ke EMR Anda.", state="complete", expanded=False)
                st.toast("Inferensi selesai dan data disimpan ke akun Anda.", icon="✅")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: RIWAYAT KLINIS (EMR)
# ============================================================
elif menu == "Riwayat Klinis (EMR)":
    st.markdown("<h2>Database EMR Privat</h2>", unsafe_allow_html=True)
    st.markdown("Arsip historis deteksi klinis yang terhubung secara eksklusif dengan akun Anda.")
    
    df_log = load_user_logs(st.session_state.active_user)

    if df_log.empty:
        st.info("Log pemeriksaan klinis pada akun ini masih kosong.")
    else:
        with st.expander("Filter Pencarian", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                opts = sorted(df_log["lesi_terdeteksi"].dropna().unique().tolist())
                selected_lesi = st.multiselect("Klasifikasi Lesi", opts, default=opts)
            with col_f2:
                tgl_series = pd.to_datetime(df_log["tanggal"], errors="coerce").dropna()
                date_range = st.date_input("Rentang Pemeriksaan", value=(tgl_series.min(), tgl_series.max())) if not tgl_series.empty else None
            with col_f3:
                min_conf = st.slider("Batas Minimum Confidence (%)", 0, 100, 0)

        mask = df_log["lesi_terdeteksi"].isin(selected_lesi) & (pd.to_numeric(df_log["confidence"], errors="coerce").fillna(0) >= (min_conf/100.0))
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            mask &= (pd.to_datetime(df_log["tanggal"], errors="coerce") >= pd.to_datetime(date_range[0])) & (pd.to_datetime(df_log["tanggal"], errors="coerce") <= pd.to_datetime(date_range[1]))

        df_filtered = df_log[mask]
        
        st.markdown("<div class='frost-card'>", unsafe_allow_html=True)
        # Sembunyikan kolom sistem internal dari tampilan UI
        df_display = df_filtered.drop(columns=["user_id"])
        st.dataframe(df_display, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Ekspor Format CSV", data=df_display.to_csv(index=False).encode("utf-8"), file_name=f"EMR_{st.session_state.active_user}.csv", mime="text/csv", use_container_width=True)
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer: df_display.to_excel(writer, index=False)
                st.download_button("📥 Ekspor Format Excel", data=buffer.getvalue(), file_name=f"EMR_{st.session_state.active_user}.xlsx", use_container_width=True)
            except ImportError: pass
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: ANALITIK DATA
# ============================================================
elif menu == "Analitik Data":
    st.markdown("<h2>Analitik Distribusi Gambar Anda</h2>", unsafe_allow_html=True)
    st.markdown("Mengevaluasi sebaran distribusi gambar dari dataset klinis yang telah Anda rekam di platform ini.")
    
    df_log = load_user_logs(st.session_state.active_user)
    
    if df_log.empty: 
        st.info("Belum ada data analitik pada akun Anda.")
    else:
        tot = len(df_log)
        df_log["confidence"] = pd.to_numeric(df_log["confidence"], errors="coerce")
        avg_conf = df_log['confidence'].mean() * 100
        df_valid = df_log[df_log["lesi_terdeteksi"] != "Tidak terdeteksi"]
        top_lesion = df_valid["lesi_terdeteksi"].mode()[0] if not df_valid.empty else "N/A"
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class='frost-card' style='text-align: center;'><p style='margin:0; font-weight:700; color:{GRAY_TEXT}; font-size:0.8rem;'>TOTAL CITRA ANDA</p><h2 style='margin:5px 0 0; color:{PRIMARY};'>{tot}</h2></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='frost-card' style='text-align: center;'><p style='margin:0; font-weight:700; color:{GRAY_TEXT}; font-size:0.8rem;'>RATA-RATA CONFIDENCE</p><h2 style='margin:5px 0 0; color:#0284c7;'>{avg_conf:.1f}%</h2></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='frost-card' style='text-align: center;'><p style='margin:0; font-weight:700; color:{GRAY_TEXT}; font-size:0.8rem;'>DISTRIBUSI TERBANYAK</p><h2 style='margin:5px 0 0; color:#b45309;'>{top_lesion.title()}</h2></div>""", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<div class='frost-card'><h4>Distribusi Gambar Berdasarkan Kelas Lesi</h4>", unsafe_allow_html=True)
            if not df_valid.empty: st.bar_chart(df_valid["lesi_terdeteksi"].value_counts(), color=PRIMARY)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_s2:
            st.markdown("<div class='frost-card'><h4>Distribusi Skala Nyeri (VAS)</h4>", unsafe_allow_html=True)
            if "s_severity" in df_log.columns and not df_log["s_severity"].dropna().empty:
                st.bar_chart(pd.to_numeric(df_log["s_severity"], errors="coerce").value_counts().sort_index(), color=ACCENT)
            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: ENSIKLOPEDIA LESI
# ============================================================
elif menu == "Ensiklopedia Lesi":
    st.markdown("<h2>Ensiklopedia Lesi Oral</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    idx = 0
    for key, info in LESION_INFO.items():
        badge_class = "badge-high" if "Tinggi" in info['urgensi'] else "badge-med" if info['urgensi'] == "Sedang" else "badge-low"
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="frost-card" style="border-top: 4px solid {PRIMARY}; height: 100%;">
                    <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:10px;">
                        <h4 style="margin:0; font-size:1.05rem;">{info['nama_klinis']}</h4>
                        <span class="badge {badge_class}" style="font-size:0.6rem;">{info['urgensi']}</span>
                    </div>
                    <p style="color:{GRAY_TEXT}; font-size:0.85rem;">{info['deskripsi']}</p>
                    <div style="background:{BG_MIST}; padding:10px; border-radius:8px; font-size:0.8rem;">
                        <strong>Tatalaksana:</strong> {info['rekomendasi']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        idx += 1


# ============================================================
# HALAMAN: PENGATURAN SISTEM
# ============================================================
elif menu == "Pengaturan Sistem":
    st.markdown("<h2>Infrastruktur & Rekomendasi</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='frost-card'>
        <h4>Spesifikasi Perangkat Lunak</h4>
        <ul>
            <li><strong>Platform:</strong> {APP_NAME}</li>
            <li><strong>Versi Aplikasi:</strong> {APP_VERSION}</li>
            <li><strong>Arsitektur Database:</strong> SQLite Relational Database</li>
            <li><strong>AI Backend:</strong> Eksekusi Dinamis (Ultralytics)</li>
        </ul>
        <hr>
        <h4>Rekomendasi Riset Digital</h4>
        <p style="color: {GRAY_TEXT}; font-size: 0.9rem;">
        Untuk keperluan kalibrasi citra digital lanjutan dalam analisis radiografis atau intraoral yang lebih presisi, sangat <strong>disarankan menggunakan GIMP</strong> (GNU Image Manipulation Program) untuk pra-pemrosesan citra sebelum diinput ke dalam model analisis <em>Computer Vision</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)
