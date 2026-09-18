"""
MAMMOUTH — My Assistant in Mouth Health
=======================================
Platform Skrining Kesehatan Rongga Mulut Berbasis Computer Vision.
Versi: Independent Multi-Tenant Edition
"""

import base64
import hashlib
import hmac
import io
import json
import os
import random
import secrets
import sqlite3
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    import altair as alt
except ImportError:
    alt = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
# 1. KONFIGURASI PROYEK INDEPENDEN & DIREKTORI
# ============================================================
APP_NAME = "MAMMOUTH"
APP_TAGLINE = "My Assistant in Mouth Health"

DATA_DIR = Path("mammouth_data")
DB_FILE = DATA_DIR / "mammouth.db"
IMG_DIR = DATA_DIR / "images"

# Dukungan arsitektur model sesuai standar riset (YOLOv8, YOLOv11, YOLOv12)
MODEL_FILES = {
    "YOLOv8": ["yolov8_best.pt", "best.pt"],
    "YOLOv11": ["yolov11_best.pt", "best.pt"],
    "YOLOv12": ["yolov12_best.pt", "best.pt"],
}

URGENCY_RANK = {"Rendah": 1, "Sedang": 2, "Sedang–Tinggi": 3, "Tinggi": 4}
RANK_URGENCY = {v: k for k, v in URGENCY_RANK.items()}

LESION_INFO: dict[str, dict[str, Any]] = {
    "karies": {"nama_klinis": "Karies", "urgensi": "Sedang–Tinggi", "tatalaksana": "Restorasi atau perawatan saluran akar sesuai kedalaman."},
    "stain calculus": {"nama_klinis": "Kalkulus & Stain", "urgensi": "Sedang", "tatalaksana": "Scaling dan root planing (SRP)."},
    "ulkus traumatikus": {"nama_klinis": "Ulkus Traumatikus", "urgensi": "Sedang", "tatalaksana": "Eliminasi faktor kausatif, obat kumur antiseptik."},
    "cheek biting": {"nama_klinis": "Morsicatio Buccarum", "urgensi": "Rendah", "tatalaksana": "Edukasi penghentian kebiasaan buruk."},
    "coated tongue": {"nama_klinis": "Coated Tongue", "urgensi": "Rendah", "tatalaksana": "Pembersihan mekanis dengan tongue scraper."},
    "linea alba": {"nama_klinis": "Linea Alba", "urgensi": "Rendah", "tatalaksana": "Varian normal, tidak memerlukan terapi."},
    "lingual varicosites": {"nama_klinis": "Lingual Varicosities", "urgensi": "Rendah", "tatalaksana": "Observasi, perubahan terkait usia."},
    "torus": {"nama_klinis": "Torus", "urgensi": "Rendah", "tatalaksana": "Observasi, bedah jika mengganggu fungsi."},
}

# ============================================================
# 2. TEMA DESAIN (KLINIK MODERN INDEPENDEN)
# ============================================================
CSS_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F8FAFC; }
.card { background: #FFFFFF; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0; margin-bottom: 20px; }
.kpi-box { text-align: center; padding: 15px; border-radius: 10px; background: #F1F5F9; border: 1px solid #E2E8F0; }
.kpi-val { font-size: 2rem; font-weight: 700; color: #3B82F6; }
.kpi-label { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
.badge { padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; color: white; }
.b-tinggi { background-color: #EF4444; }
.b-sedang { background-color: #F59E0B; }
.b-rendah { background-color: #10B981; }
.stButton > button { border-radius: 8px; font-weight: 600; }
.stButton > button[kind="primary"] { background-color: #3B82F6; border: none; }
</style>
"""

# ============================================================
# 3. LAPISAN DATABASE & KEAMANAN MULTI-TENANT
# ============================================================
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def get_conn():
    ensure_dirs()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
    pw_salt TEXT NOT NULL, full_name TEXT NOT NULL, created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, rm_code TEXT NOT NULL, name TEXT NOT NULL,
    age INTEGER, gender TEXT, contact TEXT, med_history TEXT, created_at TEXT NOT NULL,
    UNIQUE(user_id, rm_code), FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, patient_id TEXT, exam_date TEXT NOT NULL,
    model_version TEXT, image_path TEXT, annot_path TEXT, synthesis TEXT, severity INTEGER DEFAULT 0,
    urgency TEXT, is_demo INTEGER DEFAULT 0, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY, exam_id TEXT NOT NULL, user_id TEXT NOT NULL, label TEXT NOT NULL, 
    confidence REAL NOT NULL, FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE
);
"""

def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

# Keamanan Akun
def hash_password(password: str, salt: Optional[str] = None):
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200000)
    return dk.hex(), salt

def register_user(username: str, password: str, full_name: str):
    pw_hash, salt = hash_password(password)
    conn = get_conn()
    try:
        conn.execute("INSERT INTO users(id, username, pw_hash, pw_salt, full_name, created_at) VALUES(?,?,?,?,?,?)",
                     (uuid.uuid4().hex, username.lower(), pw_hash, salt, full_name, datetime.now().isoformat()))
        conn.commit()
        success = True
        msg = "Akun berhasil dibuat! Silakan masuk."
    except sqlite3.IntegrityError:
        success = False
        msg = "Username sudah digunakan. Pilih yang lain."
    conn.close()
    return success, msg

def verify_login(username: str, password: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username.lower(),)).fetchone()
    conn.close()
    if not row: return None
    calc_hash, _ = hash_password(password, row["pw_salt"])
    if hmac.compare_digest(calc_hash, row["pw_hash"]): return dict(row)
    return None

# ============================================================
# 4. MANAJEMEN DATA TERISOLASI (USER-SPECIFIC)
# ============================================================
def save_patient(user_id: str, name: str, age: int, gender: str, contact: str, history: str):
    conn = get_conn()
    pid = uuid.uuid4().hex
    rm_code = f"RM-{secrets.token_hex(2).upper()}"
    conn.execute("INSERT INTO patients VALUES(?,?,?,?,?,?,?,?,?)",
                 (pid, user_id, rm_code, name, age, gender, contact, history, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return pid

def get_patients(user_id: str):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM patients WHERE user_id=? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    return df

def save_exam(user_id: str, patient_id: str, model_v: str, img: Image.Image, annot: Image.Image, dets: list, synth: str, sev: int, is_demo: int):
    exam_id = uuid.uuid4().hex
    img_path = IMG_DIR / f"{exam_id}_raw.jpg"
    annot_path = IMG_DIR / f"{exam_id}_annot.jpg"
    
    img.convert("RGB").save(img_path, "JPEG", quality=85)
    annot.convert("RGB").save(annot_path, "JPEG", quality=85)

    urgency = "Rendah"
    if dets:
        ranks = [URGENCY_RANK.get(LESION_INFO.get(d["label"], {}).get("urgensi", "Rendah"), 1) for d in dets]
        urgency = RANK_URGENCY[max(ranks)]
    
    conn = get_conn()
    conn.execute("INSERT INTO exams VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                 (exam_id, user_id, patient_id, datetime.now().isoformat(), model_v, str(img_path), str(annot_path), synth, sev, urgency, is_demo))
    
    for d in dets:
        conn.execute("INSERT INTO detections VALUES(?,?,?,?,?)",
                     (uuid.uuid4().hex, exam_id, user_id, d["label"], float(d["confidence"])))
    conn.commit()
    conn.close()

def get_exams(user_id: str):
    conn = get_conn()
    q = """SELECT e.*, p.name as patient_name, p.rm_code, 
           (SELECT GROUP_CONCAT(d.label, ', ') FROM detections d WHERE d.exam_id = e.id) as labels 
           FROM exams e LEFT JOIN patients p ON e.patient_id = p.id WHERE e.user_id=?"""
    df = pd.read_sql(q, conn, params=(user_id,))
    conn.close()
    return df

# ============================================================
# 5. CORE AI & SINTESIS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(version: str):
    for weight in MODEL_FILES.get(version, []):
        if Path(weight).exists() and YOLO is not None:
            return YOLO(weight)
    return None

def process_inference(model, image: Image.Image, conf_thr: float):
    if model is None: # Mode Demo Fallback
        rng = random.Random()
        labels = random.sample(list(LESION_INFO.keys()), k=random.randint(0, 2))
        dets = [{"label": lbl, "confidence": round(random.uniform(conf_thr, 0.95), 2)} for lbl in labels]
        return dets, image.copy(), True
    
    res = model(image, conf=conf_thr, verbose=False)[0]
    plotted = res.plot()
    annotated = Image.fromarray(plotted[:, :, ::-1])
    dets = [{"label": res.names[int(b.cls[0].item())], "confidence": float(b.conf[0].item())} for b in res.boxes]
    return dets, annotated, False

def synthesize_clinical(dets: list, severity: int):
    if not dets:
        if severity >= 4: return f"Tidak ada anomali visual. Nyeri VAS {severity}/10 mengindikasikan masalah pulpa/periodontal tersembunyi."
        return "Tidak ada anomali visual terdeteksi. Hasil negatif tidak menyingkirkan penyakit sepenuhnya."
    
    synthesis = []
    for d in dets:
        info = LESION_INFO.get(d["label"], {})
        nama = info.get("nama_klinis", d["label"].title())
        tatalaksana = info.get("tatalaksana", "Observasi lanjut")
        synthesis.append(f"{nama} ({d['confidence']*100:.0f}%): {tatalaksana}")
    return " | ".join(synthesis)

# ============================================================
# 6. ANTARMUKA PENGGUNA (UI)
# ============================================================
st.set_page_config(page_title=APP_NAME, page_icon="🦷", layout="wide")
init_db()
st.markdown(CSS_BASE, unsafe_allow_html=True)

if "user" not in st.session_state:
    st.session_state.user = None

def ui_auth():
    st.markdown(f"<h1 style='text-align:center; color:#3B82F6;'>{APP_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#64748B;'>{APP_TAGLINE} — Ruang Kerja Independen</p><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Masuk (Login)", "Buat Akun Baru"])
        with tab1:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Masuk", type="primary", use_container_width=True):
                    user_data = verify_login(u, p)
                    if user_data:
                        st.session_state.user = user_data
                        st.rerun()
                    else:
                        st.error("Kredensial tidak valid.")
        with tab2:
            with st.form("reg_form"):
                fn = st.text_input("Nama Lengkap Panggilan (Cth: drg. Budi)")
                nu = st.text_input("Username Baru")
                np = st.text_input("Password Baru", type="password")
                if st.form_submit_button("Daftar", type="primary", use_container_width=True):
                    if len(nu) < 4 or len(np) < 6:
                        st.warning("Username min 4 karakter, Password min 6 karakter.")
                    else:
                        success, msg = register_user(nu, np, fn)
                        if success: st.success(msg)
                        else: st.error(msg)

def ui_main():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{APP_NAME} Workspace**<br><small>Dr. {user['full_name']}</small>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("Navigasi", ["Dashboard", "Manajemen Pasien", "Skrining AI Baru", "Rekam Medis"])
        st.markdown("---")
        if st.button("Keluar (Logout)", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    if menu == "Dashboard":
        st.title("Distribusi Data & Dashboard")
        exams = get_exams(user["id"])
        
        if exams.empty:
            st.info("Belum ada data pemeriksaan di akun Anda.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='kpi-box'><div class='kpi-val'>{len(exams)}</div><div class='kpi-label'>Total Skrining</div></div>", unsafe_allow_html=True)
            high_risk = len(exams[exams["urgency"].isin(["Tinggi", "Sedang–Tinggi"])])
            c2.markdown(f"<div class='kpi-box'><div class='kpi-val'>{high_risk}</div><div class='kpi-label'>Prioritas Lanjut</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='kpi-box'><div class='kpi-val'>{len(exams['patient_id'].unique()) - 1}</div><div class='kpi-label'>Pasien Terdaftar</div></div>", unsafe_allow_html=True)
            
            st.markdown("### Distribusi Citra Berdasarkan Temuan")
            st.caption("Visualisasi ini menampilkan distribusi gambar berdasarkan kelas objek yang terdeteksi, berfokus pada sebaran citra dataset lokal Anda.")
            
            # Ekstrak label dari comma-separated string ke list panjang
            all_labels = exams['labels'].dropna().str.split(', ').explode()
            if not all_labels.empty and alt is not None:
                chart_data = all_labels.value_counts().reset_index()
                chart_data.columns = ['Kelas', 'Jumlah Citra']
                chart = alt.Chart(chart_data).mark_bar(color='#3B82F6', cornerRadiusEnd=4).encode(
                    x=alt.X('Jumlah Citra:Q', title='Jumlah Distribusi Citra'),
                    y=alt.Y('Kelas:N', sort='-x', title=None),
                    tooltip=['Kelas', 'Jumlah Citra']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

    elif menu == "Manajemen Pasien":
        st.title("Database Pasien")
        with st.expander("➕ Tambah Pasien Baru", expanded=False):
            with st.form("new_patient"):
                c1, c2 = st.columns(2)
                nm = c1.text_input("Nama Pasien")
                ag = c2.number_input("Usia", min_value=1, max_value=120, value=30)
                gd = c1.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                ct = c2.text_input("Kontak (HP)")
                hx = st.text_area("Riwayat Penyakit Sistemik")
                if st.form_submit_button("Simpan Pasien", type="primary"):
                    save_patient(user["id"], nm, ag, gd, ct, hx)
                    st.success("Pasien berhasil disimpan.")
                    st.rerun()
        
        df_p = get_patients(user["id"])
        if not df_p.empty:
            st.dataframe(df_p[['rm_code', 'name', 'age', 'gender', 'contact', 'created_at']], use_container_width=True)

    elif menu == "Skrining AI Baru":
        st.title("Skrining AI Terintegrasi")
        df_p = get_patients(user["id"])
        pat_dict = {f"{r['name']} ({r['rm_code']})": r["id"] for _, r in df_p.iterrows()}
        pat_dict["Tamu / Tanpa Identitas"] = "GUEST"
        
        sel_pat = st.selectbox("Pilih Pasien", list(pat_dict.keys()))
        patient_id = pat_dict[sel_pat]

        c1, c2 = st.columns(2)
        model_ver = c1.selectbox("Arsitektur YOLO", ["YOLOv8", "YOLOv11", "YOLOv12"])
        conf = c2.slider("Ambang Batas Keyakinan (Confidence)", 0.1, 1.0, 0.25)
        
        sev = st.slider("Skala Nyeri Pasien (VAS 0-10) - Anamnesis", 0, 10, 0)
        
        file = st.file_uploader("Unggah Citra Intraoral", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file)
            st.image(img, width=400, caption="Citra Asli")
            if st.button("Jalankan Inferensi AI", type="primary"):
                with st.spinner("Memproses..."):
                    model = load_model(model_ver)
                    dets, annot, is_demo = process_inference(model, img, conf)
                    synth = synthesize_clinical(dets, sev)
                    
                    save_exam(user["id"], patient_id, model_ver, img, annot, dets, synth, sev, 1 if is_demo else 0)
                    
                    st.success("Analisis selesai dan tersimpan ke rekam medis Anda.")
                    if is_demo: st.warning("Mode Demo aktif: Bobot .pt tidak ditemukan, simulasi kotak deteksi dijalankan.")
                    
                    st.image(annot, width=400, caption="Hasil Deteksi Model")
                    st.info(f"**Sintesis Klinis:**\n{synth}")

    elif menu == "Rekam Medis":
        st.title("Arsip Rekam Medis")
        exams = get_exams(user["id"])
        if not exams.empty:
            for _, ex in exams.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="card">
                        <h4>{ex['patient_name'] or 'Tamu'} <span style="font-size:0.9rem; color:gray;">({ex['rm_code'] or '-'})</span></h4>
                        <p><strong>Tanggal:</strong> {ex['exam_date'][:16]} | <strong>Nyeri:</strong> {ex['severity']}/10 | <strong>Prioritas:</strong> {ex['urgency']}</p>
                        <p><strong>Temuan:</strong> {ex['labels'] or 'Negatif'}</p>
                        <p><strong>Sintesis:</strong> {ex['synthesis']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Belum ada rekam medis.")

if st.session_state.user:
    ui_main()
else:
    ui_auth()
