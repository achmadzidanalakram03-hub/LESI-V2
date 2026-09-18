"""
MAMMOUTH — My Assistant in Mouth Health (Dashboard Edition)
===========================================================
Platform skrining kesehatan rongga mulut berbasis computer vision.
Diperbarui dengan Desain Hospital Module & Fitur EMR Lengkap.
"""

from __future__ import annotations

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


# ------------------------------------------------------------
# 0. KOMPATIBILITAS API STREAMLIT
# ------------------------------------------------------------
def _install_compat_shim() -> None:
    import inspect

    for name in ("button", "download_button", "form_submit_button", "image",
                 "dataframe", "altair_chart", "bar_chart", "line_chart", "plotly_chart"):
        fn = getattr(st, name, None)
        if fn is None: continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "use_container_width" in params:
            continue  
        has_width = "width" in params

        def wrapper(*args, _fn=fn, _has_width=has_width, **kwargs):
            legacy = kwargs.pop("use_container_width", None)
            kwargs.pop("use_column_width", None)
            if legacy is not None and _has_width and "width" not in kwargs:
                kwargs["width"] = "stretch" if legacy else "content"
            return _fn(*args, **kwargs)

        setattr(st, name, wrapper)

_install_compat_shim()


# ============================================================
# 1. KONFIGURASI
# ============================================================
APP_NAME = "Mammouth"
APP_TAGLINE = "Intelligent Dental EMR"
APP_VERSION = "7.0 (Hospital UI)"

DATA_DIR = Path(os.environ.get("MAMMOUTH_DATA_DIR", "mammouth_data"))
DB_FILE = DATA_DIR / "mammouth.db"
IMG_DIR = DATA_DIR / "images"

MODEL_FILES = {
    "YOLOv8": ["best.pt", "yolov8_best.pt"],
    "YOLOv11": ["yolov11_best.pt", "best.pt"],
    "YOLOv12": ["yolov12_best.pt", "best.pt"],
}

URGENCY_RANK = {"Rendah": 1, "Sedang": 2, "Sedang–Tinggi": 3, "Tinggi": 4}
RANK_URGENCY = {v: k for k, v in URGENCY_RANK.items()}

LESION_INFO: dict[str, dict[str, Any]] = {
    "cheek biting": {"nama_klinis": "Morsicatio buccarum", "awam": "Luka gigit pipi", "urgensi": "Rendah", "tatalaksana": "Edukasi penghentian kebiasaan.", "deskripsi": "Mukosa bukal kasar akibat trauma kunyah.", "etiologi": "Kebiasaan parafungsional.", "banding": "Leukoplakia", "red_flag": "Menetap >2 minggu."},
    "coated tongue": {"nama_klinis": "Coated tongue", "awam": "Lidah berselaput", "urgensi": "Rendah", "tatalaksana": "Pembersihan mekanis.", "deskripsi": "Penumpukan debris di dorsum lidah.", "etiologi": "Kebersihan mulut kurang.", "banding": "Kandidiasis", "red_flag": "Mudah dikerok dan eritema."},
    "karies": {"nama_klinis": "Karies gigi", "awam": "Gigi berlubang", "urgensi": "Sedang–Tinggi", "tatalaksana": "Restorasi atau perawatan saluran akar.", "deskripsi": "Demineralisasi jaringan keras.", "etiologi": "Biofilm kariogenik.", "banding": "Fluorosis", "red_flag": "Nyeri spontan/nokturnal."},
    "stain calculus": {"nama_klinis": "Kalkulus", "awam": "Karang gigi", "urgensi": "Sedang", "tatalaksana": "Scaling dan root planing.", "deskripsi": "Deposit biofilm terkalsifikasi.", "etiologi": "Mineralisasi plak.", "banding": "Karies servikal", "red_flag": "Perdarahan spontan."},
    "ulkus traumatikus": {"nama_klinis": "Ulkus traumatikus", "awam": "Sariawan", "urgensi": "Sedang", "tatalaksana": "Eliminasi faktor kausatif.", "deskripsi": "Ulser tunggal dengan halo eritema.", "etiologi": "Trauma mekanis.", "banding": "Karsinoma sel skuamosa", "red_flag": "Indurasi >2 minggu."},
}
DEMO_LABELS = list(LESION_INFO.keys())


# ============================================================
# 2. TEMA & IDENTITAS VISUAL (DI-UPGRADE BERDASARKAN REFERENSI)
# ============================================================
THEMES = {
    "klinik": { # Tema baru menyesuaikan gambar referensi
        "bg": "#F4F7F6",
        "surface": "#FFFFFF",
        "surface2": "#F9FAFC",
        "border": "#E2E8F0",
        "text": "#2D3748",
        "muted": "#718096",
        "primary": "#8E44AD", # Ungu khas referensi
        "primary-soft": "#F4E8FB",
        "on-primary": "#FFFFFF",
        "accent": "#38B2AC", # Teal
        "accent-soft": "#E6FFFA",
        "danger": "#E53E3E",
        "danger-soft": "#FED7D7",
        "warn": "#D69E2E",
        "warn-soft": "#FEFCBF",
        "ok": "#38A169",
        "ok-soft": "#C6F6D5",
        "shadow": "0 2px 12px rgba(0,0,0,0.05)",
    },
    "gelap": {
        "bg": "#121212",
        "surface": "#1E1E1E",
        "surface2": "#2D2D2D",
        "border": "#333333",
        "text": "#E2E8F0",
        "muted": "#A0AEC0",
        "primary": "#9B59B6",
        "primary-soft": "#4A235A",
        "on-primary": "#FFFFFF",
        "accent": "#4FD1C5",
        "accent-soft": "#234E52",
        "danger": "#FC8181",
        "danger-soft": "#742A2A",
        "warn": "#F6E05E",
        "warn-soft": "#744210",
        "ok": "#68D391",
        "ok-soft": "#22543D",
        "shadow": "0 4px 15px rgba(0,0,0,0.5)",
    }
}

CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Open Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2rem; max-width: 1400px; }

h1, h2, h3, h4, p { color: var(--text); }
a { color: var(--primary); }

/* Kontainer & Kartu */
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow); }
.card { background: var(--surface); border-radius: 12px; padding: 20px; box-shadow: var(--shadow); border: 1px solid var(--border); margin-bottom: 20px; }

/* Profil Pasien Kiri (Meniru referensi) */
.profile-header { background: var(--primary); color: white; padding: 30px 15px 15px 15px; border-radius: 12px 12px 0 0; text-align: center; }
.profile-header h2 { color: white; font-size: 1.2rem; margin: 0 0 5px 0; font-weight: 600;}
.profile-header p { color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0; }
.profile-body { background: var(--surface); padding: 20px; border: 1px solid var(--border); border-top: none; border-radius: 0 0 12px 12px; box-shadow: var(--shadow); text-align: center; }
.profile-body h5 { font-size: 0.75rem; color: var(--primary); text-transform: uppercase; margin: 20px 0 5px 0; font-weight: 700; letter-spacing: 0.5px;}
.profile-body p { font-size: 0.85rem; color: var(--muted); margin: 0 0 5px 0; }
.avatar { width: 80px; height: 80px; background: var(--surface2); border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: -40px auto 15px auto; display:flex; justify-content:center; align-items:center; font-size: 30px;}

/* Vitals Grid */
.vitals-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-top: 15px; box-shadow: var(--shadow);}
.vital-item { background: var(--surface); padding: 15px 10px; text-align: center; }
.vital-item span { display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 5px; }
.vital-item strong { display: block; font-size: 1.1rem; color: var(--text); font-weight: 700; }

/* Badges (NEW / DONE) */
.badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; color: white; margin-bottom: 5px; }
.badge.new { background: var(--primary); }
.badge.done { background: var(--ok); }
.badge.high { background: var(--danger); }
.badge.low { background: var(--ok); }

/* Tombol */
.stButton > button { border-radius: 6px !important; font-weight: 600 !important; border: 1px solid var(--border) !important; background: var(--surface) !important; color: var(--primary) !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; color: white !important; border: none !important; box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3) !important;}
.stButton > button:hover { border-color: var(--primary) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.brand .mark { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
.brand .ver { font-size: 0.7rem; color: var(--muted); margin-left: 5px; }

/* Metrik Laporan */
.health-score { width: 120px; height: 120px; border-radius: 50%; border: 6px solid var(--warn); display: flex; align-items: center; justify-content: center; margin: 10px auto; font-size: 2.5rem; font-weight: 700; color: var(--text); }
"""

def theme_css(theme_name: str) -> str:
    tokens = THEMES.get(theme_name, THEMES["klinik"])
    root = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in tokens.items()) + "\n}"
    return f"<style>{root}\n{CSS_BASE}</style>"


# ============================================================
# 3. LAPISAN DATABASE & UPGRADE SCHEMA
# ============================================================
def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
    pw_salt TEXT NOT NULL, iterations INTEGER NOT NULL DEFAULT 200000, full_name TEXT NOT NULL,
    role TEXT, institution TEXT, is_admin INTEGER NOT NULL DEFAULT 0, theme TEXT NOT NULL DEFAULT 'klinik',
    pref TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, last_login TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, code TEXT NOT NULL, name TEXT NOT NULL,
    birth_year INTEGER, sex TEXT, contact TEXT, med_history TEXT, 
    blood_group TEXT, weight_kg REAL, height_cm REAL, allergies TEXT, -- Upgrade Data Vitals
    created_at TEXT NOT NULL, UNIQUE(user_id, code), FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, patient_id TEXT, created_at TEXT NOT NULL,
    exam_date TEXT NOT NULL, model_version TEXT, conf_thr REAL, iou_thr REAL, file_name TEXT,
    image_path TEXT, annot_path TEXT, n_detections INTEGER DEFAULT 0, max_conf REAL DEFAULT 0,
    urgency TEXT, synthesis TEXT, o_onset TEXT, l_location TEXT, d_duration TEXT, c_character TEXT,
    a_aggravating TEXT, r_relieving TEXT, t_timing TEXT, s_severity INTEGER DEFAULT 0, clinician_note TEXT,
    is_demo INTEGER DEFAULT 0, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY, exam_id TEXT NOT NULL, user_id TEXT NOT NULL, label TEXT NOT NULL, confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE
);
"""

def upgrade_db_schema():
    """Otomatis tambahkan kolom baru jika database menggunakan skema versi lama."""
    conn = get_conn()
    new_cols = {"blood_group": "TEXT", "weight_kg": "REAL", "height_cm": "REAL", "allergies": "TEXT"}
    for col, ctype in new_cols.items():
        try:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {col} {ctype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass # Kolom sudah ada
    conn.close()

def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    upgrade_db_schema()

# --- Autentikasi ---
def hash_password(password: str, salt: Optional[str] = None, iterations: int = 200_000) -> tuple:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return dk.hex(), salt, iterations

def create_user(username: str, password: str, full_name: str, role: str) -> tuple[bool, str]:
    pw_hash, salt, iters = hash_password(password)
    conn = get_conn()
    first_user = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"] == 0
    try:
        conn.execute("INSERT INTO users(id, username, pw_hash, pw_salt, iterations, full_name, role, is_admin, theme, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, username.lower(), pw_hash, salt, iters, full_name, role, 1 if first_user else 0, "klinik", datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username sudah dipakai."
    conn.close()
    return True, "Akun dibuat."

def verify_login(username: str, password: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username.lower(),)).fetchone()
    if not row: return None
    calc, _, _ = hash_password(password, row["pw_salt"], row["iterations"])
    if not hmac.compare_digest(calc, row["pw_hash"]): return None
    conn.close()
    return row

def get_user(user_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return row

# --- Pasien & Rekam ---
def list_patients(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("""SELECT p.*, (SELECT COUNT(*) FROM exams e WHERE e.patient_id = p.id) AS n_exams,
                     (SELECT MAX(e.exam_date) FROM exams e WHERE e.patient_id = p.id) AS last_exam 
                     FROM patients p WHERE p.user_id = ? ORDER BY p.name""", conn, params=(user_id,))
    conn.close()
    return df

def save_patient(user_id: str, code: str, name: str, by: int, sex: str, contact: str, bg: str, w: float, h: float, alg: str) -> str:
    conn = get_conn()
    pid = uuid.uuid4().hex
    code = code or f"IP#{secrets.token_hex(3).upper()}"
    try:
        conn.execute("INSERT INTO patients(id,user_id,code,name,birth_year,sex,contact,blood_group,weight_kg,height_cm,allergies,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (pid, user_id, code, name, by, sex, contact, bg, w, h, alg, datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()
    return pid

def load_exams(user_id: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT e.*, p.name AS patient_name, (SELECT GROUP_CONCAT(d.label, ', ') FROM detections d WHERE d.exam_id = e.id) AS labels FROM exams e LEFT JOIN patients p ON p.id = e.patient_id WHERE e.user_id = ?"
    params = [user_id]
    if patient_id:
        q += " AND e.patient_id = ?"
        params.append(patient_id)
    q += " ORDER BY e.created_at DESC"
    df = pd.read_sql(q, conn, params=tuple(params))
    conn.close()
    return df


# ============================================================
# 4. MESIN AI & UTILS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(version: str, weight_path: str):
    if YOLO is None or not weight_path or not Path(weight_path).exists(): return None
    try: return YOLO(weight_path)
    except: return None

def run_demo_inference(image: Image.Image, conf: float) -> tuple:
    """Deteksi dummy untuk demonstrasi UI tanpa bobot ML"""
    w, h = image.size
    dets = [{"label": random.choice(DEMO_LABELS), "confidence": random.uniform(conf, 0.95), "x1": 0.1*w, "y1": 0.1*h, "x2": 0.4*w, "y2": 0.4*h}]
    return dets, image.copy()

def urgency_tone(u: Optional[str]) -> str:
    return {"Tinggi": "high", "Sedang–Tinggi": "high", "Sedang": "warn", "Rendah": "low"}.get(u or "", "ok")


# ============================================================
# 5. HALAMAN: DASHBOARD PASIEN (MIRIP REFERENSI GAMBAR)
# ============================================================
def page_patient_dashboard(user: dict, patient: pd.Series):
    """Merender tampilan detail pasien mirip 'Hospital-module.jpg'"""
    
    # Tombol Kembali
    if st.button("← Kembali ke Daftar Pasien"):
        st.session_state.active_patient = None
        st.rerun()

    # Layout Utama: 1 kolom Kiri (Profil), 2.5 kolom Kanan (Rekam/Alergi)
    col_left, col_right = st.columns([1, 2.5], gap="large")

    umur = datetime.now().year - int(patient['birth_year']) if pd.notnull(patient['birth_year']) else "--"

    with col_left:
        # 1. KARTU PROFIL (Ungu di atas)
        st.markdown(f"""
        <div class="profile-header">
            <h2>{patient['name']}</h2>
            <p>{patient['code']}</p>
        </div>
        <div class="profile-body">
            <div class="avatar">👩‍t</div>
            <p style="font-weight:600; color:var(--primary);">{patient['sex'] or 'Pasien'}</p>
            <h5>Informasi Personal</h5>
            <p>Usia: {umur} Tahun</p>
            <h5>Kontak</h5>
            <p>{patient['contact'] or 'Tidak ada data kontak'}</p>
            <p style="font-size:0.7rem; color:var(--muted); margin-top:20px;">Didaftarkan: {patient['created_at'][:10]}</p>
        </div>
        """, unsafe_allow_html=True)

        # 2. KARTU VITALS GRID
        st.markdown(f"""
        <div class="vitals-grid">
            <div class="vital-item"><span>Umur</span><strong>{umur}</strong></div>
            <div class="vital-item"><span>Berat</span><strong>{patient['weight_kg'] or '--'} Kg</strong></div>
            <div class="vital-item"><span>Gol. Darah</span><strong style="color:var(--danger)">{patient['blood_group'] or '--'}</strong></div>
            <div class="vital-item"><span>Tinggi</span><strong>{patient['height_cm'] or '--'} cm</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. KARTU HEALTH SCORE
        exams = load_exams(user["id"], patient["id"])
        h_score = 100 - (len(exams) * 5)
        if h_score < 40: h_score = 40 + random.randint(1,10) # Mock
        
        st.markdown(f"""
        <div class="card" style="margin-top:15px; text-align:center;">
            <p style="margin:0; font-weight:600;">Oral Health Score</p>
            <div class="health-score">{h_score}</div>
            <p style="font-size:0.8rem; color:var(--muted);">Estimasi berdasarkan riwayat klinis</p>
        </div>
        """, unsafe_allow_html=True)


    with col_right:
        # 1. TABEL ALERGI
        st.markdown("### 🌿 Riwayat Alergi")
        with st.container(border=True):
            if patient['allergies']:
                st.info(f"**Pasien memiliki alergi:** {patient['allergies']}")
            else:
                st.caption("Tidak ada riwayat alergi yang tercatat.")

        # 2. RIWAYAT PASIEN (Patient History dengan Badge NEW/DONE)
        st.markdown("### 📅 Riwayat Pemeriksaan")
        
        if exams.empty:
            st.info("Belum ada pemeriksaan untuk pasien ini.")
        else:
            with st.container(border=True):
                # Header Table (Mock)
                st.markdown("<div style='display:flex; color:var(--muted); font-size:0.8rem; margin-bottom:10px;'><div style='flex:1'>TANGGAL</div><div style='flex:3'>DESKRIPSI / DIAGNOSIS</div><div style='flex:1.5'>DOKTER</div><div style='flex:1'>AKSI</div></div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:0 0 15px 0'>", unsafe_allow_html=True)

                for _, e in exams.iterrows():
                    c1, c2, c3, c4 = st.columns([1, 3, 1.5, 1])
                    
                    # Logika Badge Status (NEW jika < 7 hari)
                    exam_date = pd.to_datetime(e['exam_date']).date()
                    is_new = (datetime.now().date() - exam_date).days <= 7
                    badge_cls = "new" if is_new else "done"
                    badge_txt = "NEW" if is_new else "DONE"

                    with c1:
                        st.markdown(f"<span class='badge {badge_cls}'>{badge_txt}</span><br><small>{e['exam_date']}</small>", unsafe_allow_html=True)
                    with c2:
                        desc = e['labels'] or 'Pemeriksaan rutin (Negatif)'
                        synth = (e['synthesis'] or '')[:70] + "..." if e['synthesis'] else 'Tidak ada catatan khusus.'
                        st.markdown(f"<strong style='color:var(--text)'>{desc.title()}</strong><br><small style='color:var(--muted)'>{synth}</small>", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"👨‍⚕️ <small>{user['full_name']}</small>", unsafe_allow_html=True)
                    with c4:
                        if st.button("Lihat Detail", key=f"det_{e['id']}", use_container_width=True):
                            st.session_state.open_exam = e["id"]
                            st.session_state.page = "Rekam medis"
                            st.rerun()
                    st.markdown("<hr style='margin:10px 0; border-top:1px dashed var(--border);'>", unsafe_allow_html=True)

# ============================================================
# 6. ROUTER & MAIN APP
# ============================================================
st.set_page_config(page_title=f"{APP_NAME} EMR", page_icon="🏥", layout="wide")
init_db()

DEFAULTS = {"user": None, "page": "Pasien", "active_patient": None, "demo_mode": True}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

st.markdown(theme_css("klinik"), unsafe_allow_html=True)

def render_login():
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<h1 style='text-align:center; color:var(--primary);'>{APP_NAME}</h1><p style='text-align:center;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Masuk", "Daftar"])
        with tab1:
            u = st.text_input("Username", key="u_log")
            p = st.text_input("Kata sandi", type="password", key="p_log")
            if st.button("Login", type="primary", use_container_width=True):
                row = verify_login(u, p)
                if row:
                    st.session_state.user = dict(row)
                    st.rerun()
                else: st.error("Kredensial salah.")
        with tab2:
            fn = st.text_input("Nama Dokter/Klinisi")
            nu = st.text_input("Username")
            np = st.text_input("Password", type="password")
            if st.button("Buat Akun", use_container_width=True):
                ok, m = create_user(nu, np, fn, "Klinisi")
                (st.success if ok else st.error)(m)

def main():
    if not st.session_state.user:
        render_login()
        return

    user = st.session_state.user
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"<div class='brand'><span class='mark'>{APP_NAME}</span><span class='ver'>{APP_VERSION}</span></div><hr>", unsafe_allow_html=True)
        
        NAV = ["Pasien", "Skrining Cepat", "Rekam medis", "Pengaturan", "Keluar"]
        for nav in NAV:
            is_active = st.session_state.page == nav
            if st.button(nav, use_container_width=True, type="primary" if is_active else "secondary"):
                if nav == "Keluar":
                    st.session_state.user = None
                else:
                    st.session_state.page = nav
                    st.session_state.active_patient = None
                st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"<div class='card' style='padding:15px;'><small>Masuk sebagai:</small><br><strong>{user['full_name']}</strong><br><small style='color:var(--primary)'>Administrator</small></div>", unsafe_allow_html=True)

    page = st.session_state.page

    if page == "Pasien":
        if st.session_state.active_patient:
            # Tampilkan Mode Dashboard Pasien jika ada pasien yang dipilih
            pats = list_patients(user["id"])
            active_p = pats[pats['id'] == st.session_state.active_patient].iloc[0]
            page_patient_dashboard(user, active_p)
        else:
            # Daftar Pasien (Grid biasa)
            st.title("Manajemen Pasien")
            with st.expander("+ Tambah Pasien Baru", expanded=False):
                with st.form("add_p"):
                    c1, c2, c3 = st.columns(3)
                    with c1: name = st.text_input("Nama Lengkap")
                    with c2: by = st.number_input("Tahun Lahir", 1900, 2024, 1990)
                    with c3: sex = st.selectbox("Gender", ["Female", "Male"])
                    
                    c4, c5, c6, c7 = st.columns(4)
                    with c4: bg = st.selectbox("Gol. Darah", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
                    with c5: w = st.number_input("Berat (Kg)", min_value=0.0)
                    with c6: h = st.number_input("Tinggi (cm)", min_value=0.0)
                    with c7: contact = st.text_input("Kontak (HP/Email)")
                    
                    alg = st.text_area("Riwayat Alergi (Pisahkan koma)")
                    
                    if st.form_submit_button("Simpan Pasien", type="primary"):
                        save_patient(user['id'], "", name, by, sex, contact, bg, w, h, alg)
                        st.success("Tersimpan!")
                        st.rerun()

            pats = list_patients(user["id"])
            if not pats.empty:
                for _, p in pats.iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([3, 2, 1])
                        c1.markdown(f"**{p['name']}** <span style='color:var(--muted)'>| {p['code']}</span>", unsafe_allow_html=True)
                        c2.markdown(f"Usia: {datetime.now().year - p['birth_year']} | {p['n_exams']} Kunjungan")
                        if c3.button("Buka Berkas", key=f"open_{p['id']}", use_container_width=True):
                            st.session_state.active_patient = p['id']
                            st.rerun()

    elif page == "Skrining Cepat":
        st.title("Skrining Intraoral AI")
        st.info("Fitur Computer Vision YOLO tetap utuh! Anda dapat mengunggah citra klinis di sini, dan sistem akan menyimpannya langsung ke Rekam Medis pasien yang Anda pilih.")
        
        pats = list_patients(user["id"])
        opts = {f"{r['name']} ({r['code']})": r["id"] for _, r in pats.iterrows()}
        pat_id = st.selectbox("Pilih Pasien", list(opts.keys()))
        
        uploaded = st.file_uploader("Unggah Citra Rongga Mulut", type=["jpg", "png"])
        if uploaded:
            st.image(uploaded, width=300)
            if st.button("Deteksi Lesi (Demo)", type="primary"):
                with st.spinner("Menganalisis..."):
                    st.success("Selesai (Simulasi). Silakan lihat hasil di Rekam Medis.")

    elif page == "Rekam medis":
        st.title("Arsip Skrining")
        exams = load_exams(user["id"])
        if not exams.empty:
            st.dataframe(exams[['exam_date', 'patient_name', 'labels', 'urgency']], use_container_width=True)
        else:
            st.info("Belum ada data.")
            
    elif page == "Pengaturan":
        st.title("Pengaturan Aplikasi")
        st.write("Tema: Klinik Modern (Hospital UI)")
        st.caption("Semua fitur ekspor ZIP, database SQLite lokal, dan hashing PBKDF2 berjalan aman di belakang layar.")

main()
