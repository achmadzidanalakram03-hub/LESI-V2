"""
MAMMOUTH — My Assistant in Mouth Health
=======================================
Platform skrining kesehatan rongga mulut berbasis computer vision.

Diperbarui: Desain Hospital Module (Klinik Modern) + Fitur Inti Penuh
Fitur: AI YOLOv8/v11/v12, Anamnesis OLD CARTS, Ekspor HTML/ZIP, Vitals Pasien.
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
except Exception:
    alt = None

try:
    from ultralytics import YOLO
except Exception:
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
        except (TypeError, ValueError): continue
        if "use_container_width" in params: continue  
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
APP_VERSION = "8.0"

DATA_DIR = Path(os.environ.get("MAMMOUTH_DATA_DIR", "mammouth_data"))
DB_FILE = DATA_DIR / "mammouth.db"
IMG_DIR = DATA_DIR / "images"

MODEL_FILES = {
    "YOLOv8": ["best.pt", "yolov8_best.pt"],
    "YOLOv11": ["yolov11_best.pt", "yolo11_best.pt", "best.pt"],
    "YOLOv12": ["yolov12_best.pt", "yolo12_best.pt", "best.pt"],
}

URGENCY_RANK = {"Rendah": 1, "Sedang": 2, "Sedang–Tinggi": 3, "Tinggi": 4}
RANK_URGENCY = {v: k for k, v in URGENCY_RANK.items()}

LESION_INFO: dict[str, dict[str, Any]] = {
    "cheek biting": {"nama_klinis": "Morsicatio buccarum", "awam": "Luka gigit pipi", "urgensi": "Rendah", "tatalaksana": "Edukasi penghentian kebiasaan (habit breaking). Evaluasi ulang bila menetap >2 minggu.", "deskripsi": "Mukosa bukal kasar akibat trauma kunyah berulang.", "etiologi": "Kebiasaan parafungsional, stres.", "banding": "Leukoplakia", "red_flag": "Menetap >2 minggu."},
    "coated tongue": {"nama_klinis": "Coated tongue", "awam": "Lidah berselaput", "urgensi": "Rendah", "tatalaksana": "Pembersihan mekanis dengan tongue scraper, hidrasi, DHE.", "deskripsi": "Penumpukan debris dan sisa makanan di dorsum lidah.", "etiologi": "Kebersihan mulut kurang, diet lunak.", "banding": "Kandidiasis pseudomembran", "red_flag": "Mudah dikerok dan meninggalkan dasar eritema."},
    "karies": {"nama_klinis": "Karies gigi", "awam": "Gigi berlubang", "urgensi": "Sedang–Tinggi", "tatalaksana": "Konfirmasi kedalaman dengan radiograf. Rencana restorasi atau perawatan saluran akar.", "deskripsi": "Demineralisasi jaringan keras gigi oleh asam bakteri.", "etiologi": "Biofilm kariogenik, karbohidrat, waktu.", "banding": "Stain ekstrinsik, fluorosis", "red_flag": "Nyeri spontan/nokturnal menandakan keterlibatan pulpa."},
    "linea alba": {"nama_klinis": "Linea alba buccalis", "awam": "Garis putih di pipi", "urgensi": "Rendah", "tatalaksana": "Varian normal. Tidak memerlukan terapi; cukup edukasi.", "deskripsi": "Garis keratotik putih horizontal bilateral.", "etiologi": "Friksi oklusal ringan kronis.", "banding": "Cheek biting", "red_flag": "Menebal unilateral atau ulseratif."},
    "lingual varicosites": {"nama_klinis": "Lingual varicosities", "awam": "Pelebaran vena bawah lidah", "urgensi": "Rendah", "tatalaksana": "Tidak memerlukan tindakan invasif. Edukasi sifat jinak.", "deskripsi": "Vena berkelok warna biru-keunguan pada ventral lidah.", "etiologi": "Perubahan degeneratif terkait usia.", "banding": "Hemangioma", "red_flag": "Lesi vaskular yang mudah berdarah."},
    "stain calculus": {"nama_klinis": "Stain & Kalkulus", "awam": "Karang gigi", "urgensi": "Sedang", "tatalaksana": "Scaling dan root planing, poles, kontrol berkala 6 bulan.", "deskripsi": "Deposit biofilm terkalsifikasi pada permukaan gigi.", "etiologi": "Mineralisasi plak oleh saliva.", "banding": "Karies servikal", "red_flag": "Kalkulus subgingiva dengan perdarahan (periodontitis)."},
    "torus": {"nama_klinis": "Torus palatinus/mandibularis", "awam": "Tonjolan tulang", "urgensi": "Rendah", "tatalaksana": "Observasi. Bedah hanya bila mengganggu fungsi/protesa.", "deskripsi": "Eksostosis tulang jinak berbatas tegas.", "etiologi": "Genetik dan beban oklusal.", "banding": "Osteoma, abses", "red_flag": "Pembesaran cepat atau nyeri ulserasi."},
    "ulkus traumatikus": {"nama_klinis": "Ulkus traumatikus", "awam": "Sariawan", "urgensi": "Sedang", "tatalaksana": "Eliminasi faktor kausatif, obat kumur antiseptik. Evaluasi 14 hari.", "deskripsi": "Ulser tunggal dengan halo eritema dan dasar kuning.", "etiologi": "Trauma mekanis, termal, kimiawi.", "banding": "Karsinoma sel skuamosa", "red_flag": "Indurasi tidak nyeri menetap >2 minggu (wajib biopsi)."},
}
DEMO_LABELS = list(LESION_INFO.keys())


# ============================================================
# 2. TEMA & IDENTITAS VISUAL (KLINIK MODERN)
# ============================================================
THEMES = {
    "klinik": {
        "bg": "#F4F7F6", "surface": "#FFFFFF", "surface2": "#F9FAFC", "border": "#E2E8F0",
        "text": "#2D3748", "muted": "#718096", "primary": "#8E44AD", "primary-soft": "#F4E8FB",
        "on-primary": "#FFFFFF", "accent": "#38B2AC", "accent-soft": "#E6FFFA", "danger": "#E53E3E",
        "danger-soft": "#FED7D7", "warn": "#D69E2E", "warn-soft": "#FEFCBF", "ok": "#38A169",
        "ok-soft": "#C6F6D5", "shadow": "0 2px 12px rgba(0,0,0,0.05)",
    },
    "gelap": {
        "bg": "#121212", "surface": "#1E1E1E", "surface2": "#2D2D2D", "border": "#333333",
        "text": "#E2E8F0", "muted": "#A0AEC0", "primary": "#9B59B6", "primary-soft": "#4A235A",
        "on-primary": "#FFFFFF", "accent": "#4FD1C5", "accent-soft": "#234E52", "danger": "#FC8181",
        "danger-soft": "#742A2A", "warn": "#F6E05E", "warn-soft": "#744210", "ok": "#68D391",
        "ok-soft": "#22543D", "shadow": "0 4px 15px rgba(0,0,0,0.5)",
    }
}

CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Open Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2rem; max-width: 1400px; }

h1, h2, h3, h4, p { color: var(--text); }
a { color: var(--primary); }

div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow); }
.card { background: var(--surface); border-radius: 12px; padding: 20px; box-shadow: var(--shadow); border: 1px solid var(--border); margin-bottom: 20px; }

/* Profil Pasien Kiri */
.profile-header { background: var(--primary); color: white; padding: 30px 15px 15px 15px; border-radius: 12px 12px 0 0; text-align: center; }
.profile-header h2 { color: white; font-size: 1.2rem; margin: 0 0 5px 0; font-weight: 600;}
.profile-header p { color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0; }
.profile-body { background: var(--surface); padding: 20px; border: 1px solid var(--border); border-top: none; border-radius: 0 0 12px 12px; box-shadow: var(--shadow); text-align: center; }
.profile-body h5 { font-size: 0.75rem; color: var(--primary); text-transform: uppercase; margin: 20px 0 5px 0; font-weight: 700; letter-spacing: 0.5px;}
.profile-body p { font-size: 0.85rem; color: var(--muted); margin: 0 0 5px 0; }
.avatar { width: 80px; height: 80px; background: var(--surface2); border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin: -40px auto 15px auto; display:flex; justify-content:center; align-items:center; font-size: 30px;}

.vitals-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-top: 15px; box-shadow: var(--shadow);}
.vital-item { background: var(--surface); padding: 15px 10px; text-align: center; }
.vital-item span { display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 5px; }
.vital-item strong { display: block; font-size: 1.1rem; color: var(--text); font-weight: 700; }

.badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; color: white; margin-bottom: 5px; }
.badge.new { background: var(--primary); }
.badge.done { background: var(--ok); }

.stButton > button { border-radius: 6px !important; font-weight: 600 !important; border: 1px solid var(--border) !important; background: var(--surface) !important; color: var(--primary) !important; }
.stButton > button[kind="primary"] { background: var(--primary) !important; color: white !important; border: none !important; box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3) !important;}
.stButton > button:hover { border-color: var(--primary) !important; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.brand .mark { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
.brand .ver { font-size: 0.7rem; color: var(--muted); margin-left: 5px; }
.health-score { width: 120px; height: 120px; border-radius: 50%; border: 6px solid var(--warn); display: flex; align-items: center; justify-content: center; margin: 10px auto; font-size: 2.5rem; font-weight: 700; color: var(--text); }
"""

def theme_css(theme_name: str) -> str:
    tokens = THEMES.get(theme_name, THEMES["klinik"])
    root = ":root{\n" + "\n".join(f"  --{k}: {v};" for k, v in tokens.items()) + "\n}"
    return f"<style>{root}\n{CSS_BASE}</style>"


# ============================================================
# 3. LAPISAN DATABASE
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
    blood_group TEXT, weight_kg REAL, height_cm REAL, allergies TEXT,
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
    conn = get_conn()
    new_cols = {"blood_group": "TEXT", "weight_kg": "REAL", "height_cm": "REAL", "allergies": "TEXT"}
    for col, ctype in new_cols.items():
        try:
            conn.execute(f"ALTER TABLE patients ADD COLUMN {col} {ctype}")
            conn.commit()
        except sqlite3.OperationalError: pass
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
    conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), row["id"]))
    conn.commit()
    conn.close()
    return row

def get_user(user_id: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return row

# --- Database Pasien & Pemeriksaan ---
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
    except sqlite3.IntegrityError: pass
    conn.close()
    return pid

def save_exam(user_id: str, exam: dict, detections: list[dict]) -> str:
    exam_id = exam.get("id") or uuid.uuid4().hex
    conn = get_conn()
    conn.execute(
        """INSERT INTO exams(id,user_id,patient_id,created_at,exam_date,model_version,conf_thr,iou_thr,
                             file_name,image_path,annot_path,n_detections,max_conf,urgency,synthesis,
                             o_onset,l_location,d_duration,c_character,a_aggravating,r_relieving,t_timing,
                             s_severity,clinician_note,is_demo)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (exam_id, user_id, exam.get("patient_id"), datetime.now().isoformat(), exam.get("exam_date"),
         exam.get("model_version"), exam.get("conf_thr"), exam.get("iou_thr"), exam.get("file_name"),
         exam.get("image_path"), exam.get("annot_path"), len(detections), exam.get("max_conf", 0.0),
         exam.get("urgency"), exam.get("synthesis"), exam.get("o_onset", "-"), exam.get("l_location", "-"),
         exam.get("d_duration", "-"), exam.get("c_character", "-"), exam.get("a_aggravating", "-"),
         exam.get("r_relieving", "-"), exam.get("t_timing", "-"), int(exam.get("s_severity", 0) or 0),
         exam.get("clinician_note", ""), int(exam.get("is_demo", 0)))
    )
    for d in detections:
        conn.execute("INSERT INTO detections(id,exam_id,user_id,label,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, exam_id, user_id, d["label"], float(d["confidence"]), d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")))
    conn.commit()
    conn.close()
    return exam_id

def load_exams(user_id: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT e.*, p.name AS patient_name, p.code AS patient_code, (SELECT GROUP_CONCAT(d.label, ', ') FROM detections d WHERE d.exam_id = e.id) AS labels FROM exams e LEFT JOIN patients p ON p.id = e.patient_id WHERE e.user_id = ?"
    params = [user_id]
    if patient_id:
        q += " AND e.patient_id = ?"
        params.append(patient_id)
    q += " ORDER BY e.created_at DESC"
    df = pd.read_sql(q, conn, params=tuple(params))
    conn.close()
    return df

def get_exam(user_id: str, exam_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT e.*, p.name AS patient_name, p.code AS patient_code, p.birth_year, p.sex FROM exams e LEFT JOIN patients p ON p.id = e.patient_id WHERE e.id=? AND e.user_id=?", (exam_id, user_id)).fetchone()
    if not row: return None
    dets = conn.execute("SELECT label, confidence FROM detections WHERE exam_id=? AND user_id=? ORDER BY confidence DESC", (exam_id, user_id)).fetchall()
    conn.close()
    out = dict(row)
    out["detections"] = [dict(d) for d in dets]
    return out


# ============================================================
# 4. MESIN AI & UTILS SINTESIS (YOLO UTUH)
# ============================================================
def find_weights(version: str) -> Optional[Path]:
    for cand in MODEL_FILES.get(version, ["best.pt"]):
        p = Path(cand)
        if p.exists(): return p
    return None

@st.cache_resource(show_spinner=False)
def load_model(version: str, weight_path: str):
    if YOLO is None or not weight_path: return None
    try: return YOLO(weight_path)
    except: return None

def run_inference(model, image: Image.Image, conf: float, iou: float) -> tuple:
    results = model(image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    plotted = res.plot() 
    annotated = Image.fromarray(plotted[:, :, ::-1])
    dets = []
    for b in res.boxes:
        xy = b.xyxy[0].tolist()
        dets.append({"label": res.names[int(b.cls[0].item())], "confidence": float(b.conf[0].item()), "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3]})
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets, annotated

def run_demo_inference(image: Image.Image, conf: float) -> tuple:
    rng = random.Random(hash(image.tobytes()[:2048]) & 0xFFFF)
    w, h = image.size
    n = rng.choice([0, 1, 1, 2])
    dets = []
    for label in rng.sample(DEMO_LABELS, k=min(n, len(DEMO_LABELS))):
        c = round(rng.uniform(max(conf, 0.3), 0.96), 4)
        x1, y1 = rng.uniform(0.05, 0.5) * w, rng.uniform(0.05, 0.5) * h
        dets.append({"label": label, "confidence": c, "x1": x1, "y1": y1, "x2": x1 + 0.3 * w, "y2": y1 + 0.28 * h})
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets, image.copy()

def urgency_of(labels: list[str], severity: int = 0) -> str:
    rank = 0
    for lb in labels:
        rank = max(rank, URGENCY_RANK.get(LESION_INFO.get(lb.lower(), {}).get("urgensi", "Rendah"), 1))
    if severity >= 7: rank = max(rank, 3)
    elif severity >= 4: rank = max(rank, 2)
    return RANK_URGENCY.get(rank, "Rendah") if rank else "Rendah"

def synthesize(detections: list[dict], anam: dict) -> str:
    sev = int(anam.get("s_severity", 0) or 0)
    char = str(anam.get("c_character", "")).lower()
    if not detections:
        if sev >= 4: return f"Tidak ada anomali visual yang terdeteksi, namun pasien melaporkan nyeri VAS {sev}/10. Pertimbangkan sumber non-visual seperti karies tersembunyi atau masalah periodontal."
        return "Tidak ada anomali visual yang terdeteksi pada citra ini. Hasil negatif tidak menyingkirkan penyakit."
    lines = []
    for d in detections:
        key = d["label"].lower()
        info = LESION_INFO.get(key, {})
        nama = info.get("nama_klinis", d["label"])
        lines.append(f"{nama} ({d['confidence']*100:.0f}%) — {info.get('tatalaksana', 'Observasi klinis').split('.')[0]}.")
    return " ".join(f"{i}. {t}" for i, t in enumerate(lines, 1))

def enhance_img(img: Image.Image, brightness: float, contrast: float, sharpness: float, auto: bool) -> Image.Image:
    out = ImageOps.exif_transpose(img)
    if auto: out = ImageOps.autocontrast(out, cutoff=1)
    if brightness != 1.0: out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0: out = ImageEnhance.Contrast(out).enhance(contrast)
    if sharpness != 1.0: out = ImageEnhance.Sharpness(out).enhance(sharpness)
    return out

def store_image(user_id: str, exam_id: str, img: Image.Image, kind: str) -> str:
    folder = IMG_DIR / user_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{exam_id}_{kind}.jpg"
    img.convert("RGB").thumbnail((1400, 1400))
    img.save(path, "JPEG", quality=88, optimize=True)
    return str(path)

def img_to_b64(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists(): return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((900, 900))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except: return None


# ============================================================
# 5. EKSPOR LAPORAN
# ============================================================
def build_report_html(exam: dict, clinician: str, institution: str) -> str:
    b64 = img_to_b64(exam.get("annot_path") or exam.get("image_path"))
    img_block = f"<img src='data:image/jpeg;base64,{b64}'>" if b64 else "<p class='muted'>Citra tidak tersimpan.</p>"
    rows = "".join(f"<tr><td>{d['label']}</td><td>{LESION_INFO.get(d['label'].lower(), {}).get('nama_klinis', '—')}</td><td class='num'>{d['confidence']*100:.1f}%</td></tr>" for d in exam.get("detections", [])) or "<tr><td colspan='3' class='muted'>Tidak ada anomali terdeteksi.</td></tr>"
    olds = [("Onset", "o_onset"), ("Lokasi", "l_location"), ("Durasi", "d_duration"), ("Karakter", "c_character"), ("Memperberat", "a_aggravating"), ("Meredakan", "r_relieving"), ("Waktu", "t_timing")]
    anam_rows = "".join(f"<tr><th>{lab}</th><td>{exam.get(k) or '—'}</td></tr>" for lab, k in olds)
    anam_rows += f"<tr><th>Skala nyeri</th><td>{exam.get('s_severity', 0)}/10</td></tr>"
    demo_note = "<p class='warn'>Mode demo: Simulasi.</p>" if exam.get("is_demo") else ""

    return f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Laporan Skrining {exam.get('id','')[:8].upper()}</title>
<style>
  body {{ font-family: sans-serif; color: #2D3748; line-height: 1.5; max-width: 800px; margin: 0 auto; padding: 24px; }}
  header {{ border-bottom: 2px solid #8E44AD; padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
  .mark {{ font-size: 1.5rem; font-weight: bold; color: #8E44AD; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #E2E8F0; }}
  img {{ width: 100%; border-radius: 8px; margin-top: 10px; border: 1px solid #ccc; }}
  .synth {{ background: #F4E8FB; border-left: 4px solid #8E44AD; padding: 12px; border-radius: 6px; }}
</style></head><body>
<header><div><div class="mark">Mammouth EMR</div></div><div>No. {exam.get('id','')[:8].upper()}<br>{exam.get('exam_date','')}</div></header>
{demo_note}<h2>Identitas</h2><table>
  <tr><th>Pasien</th><td>{exam.get('patient_name')} ({exam.get('patient_code')})</td></tr>
  <tr><th>Pemeriksa</th><td>{clinician} {institution}</td></tr>
</table>
<h2>Citra dan temuan</h2>{img_block}
<table style="margin-top:14px"><thead><tr><th>Kelas</th><th>Diagnosis</th><th>Keyakinan</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Anamnesis OLD CARTS</h2><table>{anam_rows}</table>
<h2>Sintesis klinis</h2><div class="synth">{exam.get('synthesis')}</div>
{f"<h2>Catatan</h2><p>{exam.get('clinician_note')}</p>" if exam.get('clinician_note') else ""}
<footer><br><br>______________________________<br>{clinician}</footer></body></html>"""

def build_export_zip(user_id: str) -> bytes:
    exams = load_exams(user_id)
    pats = list_patients(user_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pemeriksaan.csv", exams.to_csv(index=False))
        z.writestr("pasien.csv", pats.to_csv(index=False))
        folder = IMG_DIR / user_id
        if folder.exists():
            for f in folder.glob("*.jpg"): z.write(f, f"citra/{f.name}")
    return buf.getvalue()


# ============================================================
# 6. TAMPILAN HALAMAN
# ============================================================
def page_patient_dashboard(user: dict, patient: pd.Series):
    if st.button("← Kembali ke Daftar Pasien"):
        st.session_state.active_patient = None
        st.rerun()

    col_left, col_right = st.columns([1, 2.5], gap="large")
    umur = datetime.now().year - int(patient['birth_year']) if pd.notnull(patient['birth_year']) else "--"

    with col_left:
        st.markdown(f"""
        <div class="profile-header"><h2>{patient['name']}</h2><p>{patient['code']}</p></div>
        <div class="profile-body"><div class="avatar">👩‍⚕️</div>
            <p style="font-weight:600; color:var(--primary);">{patient['sex'] or 'Pasien'}</p>
            <h5>Informasi Personal</h5><p>Usia: {umur} Tahun</p>
            <h5>Kontak</h5><p>{patient['contact'] or '-'}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="vitals-grid">
            <div class="vital-item"><span>Umur</span><strong>{umur}</strong></div>
            <div class="vital-item"><span>Berat</span><strong>{patient['weight_kg'] or '--'} Kg</strong></div>
            <div class="vital-item"><span>Gol. Darah</span><strong style="color:var(--danger)">{patient['blood_group'] or '--'}</strong></div>
            <div class="vital-item"><span>Tinggi</span><strong>{patient['height_cm'] or '--'} cm</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        exams = load_exams(user["id"], patient["id"])
        h_score = max(40, 100 - (len(exams) * 5))
        st.markdown(f"""
        <div class="card" style="margin-top:15px; text-align:center;">
            <p style="margin:0; font-weight:600;">Oral Health Score</p>
            <div class="health-score">{h_score}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🌿 Riwayat Alergi")
        with st.container(border=True):
            if patient['allergies']: st.info(f"**Alergi:** {patient['allergies']}")
            else: st.caption("Tidak ada riwayat alergi.")

        st.markdown("### 📅 Riwayat Pemeriksaan")
        if exams.empty: st.info("Belum ada pemeriksaan.")
        else:
            with st.container(border=True):
                for _, e in exams.iterrows():
                    c1, c2, c3 = st.columns([1, 3, 1])
                    is_new = (datetime.now().date() - pd.to_datetime(e['exam_date']).date()).days <= 7
                    badge = "<span class='badge new'>NEW</span>" if is_new else "<span class='badge done'>DONE</span>"
                    c1.markdown(f"{badge}<br><small>{e['exam_date']}</small>", unsafe_allow_html=True)
                    desc = e['labels'] or 'Negatif'
                    c2.markdown(f"**{desc.title()}**<br><small>{(e['synthesis'] or '')[:80]}...</small>", unsafe_allow_html=True)
                    if c3.button("Detail Laporan", key=f"det_{e['id']}"):
                        st.session_state.open_exam = e["id"]
                        st.session_state.page = "Rekam Medis"
                        st.rerun()
                    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

def page_screening(user: dict, model):
    st.title("Skrining AI & OLD CARTS")
    st.write("Unggah citra intraoral dan isi formulir anamnesis untuk menghasilkan rekam medis komprehensif.")

    demo = st.session_state.demo_mode and model is None
    if model is None and not demo:
        st.warning("Bobot YOLO tidak ditemukan. Masukkan file .pt ke folder atau aktifkan Mode Demo di sidebar.")
        return

    pats = list_patients(user["id"])
    opts = {f"{r['name']} ({r['code']})": r["id"] for _, r in pats.iterrows()}
    opts["— Pasien Baru / Tanpa Identitas —"] = None
    patient_id = st.selectbox("Pilih Pasien (Opsional)", list(opts.keys()), format_func=lambda x: x)
    if patient_id != "— Pasien Baru / Tanpa Identitas —": patient_id = opts[patient_id]
    else: patient_id = None

    # OLD CARTS FORM
    with st.expander("Anamnesis OLD CARTS", expanded=not st.session_state.get('anamnesis')):
        with st.form("form_anamnesis"):
            a1, a2 = st.columns(2)
            cur = st.session_state.get("anamnesis", {})
            with a1:
                o = st.text_input("Onset (Kapan muncul)", cur.get("o_onset", ""))
                l = st.text_input("Location (Lokasi)", cur.get("l_location", ""))
                d = st.text_input("Duration (Durasi nyeri)", cur.get("d_duration", ""))
                ch = st.text_input("Character (Karakteristik)", cur.get("c_character", ""))
            with a2:
                ag = st.text_input("Aggravating (Memperberat)", cur.get("a_aggravating", ""))
                rl = st.text_input("Relieving (Meredakan)", cur.get("r_relieving", ""))
                tm = st.text_input("Timing (Waktu)", cur.get("t_timing", ""))
                sv = st.slider("Severity (Skala nyeri VAS)", 0, 10, int(cur.get("s_severity", 0)))
            if st.form_submit_button("Simpan Anamnesis", type="primary"):
                st.session_state.anamnesis = {"o_onset": o, "l_location": l, "d_duration": d, "c_character": ch, "a_aggravating": ag, "r_relieving": rl, "t_timing": tm, "s_severity": sv}
                st.success("Tersimpan!")

    # CITRA
    st.markdown("### Unggah Citra Klinis")
    files = st.file_uploader("Pilih foto", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files:
        st.markdown("### Penyesuaian Citra")
        e1, e2, e3, e4 = st.columns(4)
        bright = e1.slider("Kecerahan", 0.5, 2.0, 1.0)
        contrast = e2.slider("Kontras", 0.5, 2.0, 1.0)
        sharp = e3.slider("Ketajaman", 0.5, 2.5, 1.0)
        auto = e4.toggle("Auto-kontras", value=False)

        for f in files:
            img = Image.open(f).convert("RGB")
            proc_img = enhance_img(img, bright, contrast, sharp, auto)
            st.image(proc_img, caption=f.name, width=300)

            if st.button(f"Jalankan Inferensi pada {f.name}", type="primary", key=f"run_{f.name}"):
                with st.spinner("Memproses dengan YOLO..."):
                    if demo: dets, annot = run_demo_inference(proc_img, st.session_state.conf_thr)
                    else: dets, annot = run_inference(model, proc_img, st.session_state.conf_thr, 0.45)
                    
                    st.image(annot, caption="Hasil Deteksi")
                    
                    exam = {
                        "patient_id": patient_id, "exam_date": datetime.now().strftime("%Y-%m-%d"),
                        "model_version": "DEMO" if demo else st.session_state.model_version,
                        "conf_thr": st.session_state.conf_thr, "iou_thr": 0.45, "file_name": f.name,
                        "urgency": urgency_of([d["label"] for d in dets], st.session_state.get('anamnesis',{}).get('s_severity',0)),
                        "synthesis": synthesize(dets, st.session_state.get('anamnesis',{})),
                        "is_demo": 1 if demo else 0, **st.session_state.get('anamnesis',{})
                    }
                    
                    exam_id = save_exam(user["id"], exam, dets)
                    st.success(f"Disimpan ke database dengan ID: {exam_id[:8]}")
                    
                    # Tampilkan sintesis
                    st.info(exam["synthesis"])

def page_records(user: dict):
    st.title("Rekam Medis Pasien")
    if st.session_state.open_exam:
        full = get_exam(user["id"], st.session_state.open_exam)
        if full:
            st.markdown(f"### Detail Pemeriksaan {full['exam_date']}")
            st.download_button("Unduh Laporan PDF/HTML", data=build_report_html(full, user["full_name"], ""), file_name=f"laporan_{full['id'][:8]}.html", mime="text/html")
            
            c1, c2 = st.columns(2)
            orig = IMG_DIR / user["id"] / f"{full['id']}_asli.jpg"
            if orig.exists(): c1.image(Image.open(orig), caption="Asli")
            annot = IMG_DIR / user["id"] / f"{full['id']}_anotasi.jpg"
            if annot.exists(): c2.image(Image.open(annot), caption="Anotasi")
            
            st.write(full['synthesis'])
        if st.button("Tutup Detail"):
            st.session_state.open_exam = None
            st.rerun()
        st.markdown("---")

    exams = load_exams(user["id"])
    if not exams.empty:
        st.dataframe(exams[['exam_date', 'patient_name', 'labels', 'urgency', 'file_name']], use_container_width=True)
        st.download_button("Ekspor Seluruh Database (ZIP)", build_export_zip(user["id"]), file_name="export_emr.zip", mime="application/zip")
    else:
        st.info("Belum ada riwayat.")

# ============================================================
# 7. ROUTER & MAIN APP
# ============================================================
st.set_page_config(page_title=f"{APP_NAME} EMR", page_icon="🏥", layout="wide")
init_db()

DEFAULTS = {"user": None, "page": "Dashboard Pasien", "active_patient": None, "demo_mode": False, "conf_thr": 0.25, "model_version": "YOLOv8"}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

st.markdown(theme_css("klinik"), unsafe_allow_html=True)

def render_login():
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<h1 style='text-align:center; color:var(--primary);'>{APP_NAME}</h1><p style='text-align:center;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
        u = st.text_input("Username", key="u_log")
        p = st.text_input("Kata sandi", type="password", key="p_log")
        if st.button("Login", type="primary", use_container_width=True):
            row = verify_login(u, p)
            if row:
                st.session_state.user = dict(row)
                st.rerun()
            else: st.error("Kredensial salah.")

def main():
    if not st.session_state.user:
        render_login()
        return

    user = st.session_state.user
    
    with st.sidebar:
        st.markdown(f"<div class='brand'><span class='mark'>{APP_NAME}</span><span class='ver'>{APP_VERSION}</span></div><hr>", unsafe_allow_html=True)
        
        NAV = ["Dashboard Pasien", "Skrining AI & OLD CARTS", "Rekam Medis", "Keluar"]
        for nav in NAV:
            if st.button(nav, use_container_width=True, type="primary" if st.session_state.page == nav else "secondary"):
                if nav == "Keluar": st.session_state.user = None
                else: 
                    st.session_state.page = nav
                    if nav != "Dashboard Pasien": st.session_state.active_patient = None
                st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.session_state.demo_mode = st.toggle("Mode Demo (Tanpa YOLO)", st.session_state.demo_mode)
        st.session_state.conf_thr = st.slider("Ambang Keyakinan AI", 0.05, 0.95, 0.25)

    weights = find_weights(st.session_state.model_version)
    model = load_model(st.session_state.model_version, str(weights) if weights else "")

    if st.session_state.page == "Dashboard Pasien":
        if st.session_state.active_patient:
            pats = list_patients(user["id"])
            page_patient_dashboard(user, pats[pats['id'] == st.session_state.active_patient].iloc[0])
        else:
            st.title("Manajemen Pasien")
            with st.expander("+ Pendaftaran Pasien Baru", expanded=False):
                with st.form("add_p"):
                    c1, c2, c3 = st.columns(3)
                    with c1: name = st.text_input("Nama Lengkap")
                    with c2: by = st.number_input("Tahun Lahir", 1900, 2026, 1990)
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
                        if c3.button("Buka Profil", key=f"open_{p['id']}", use_container_width=True):
                            st.session_state.active_patient = p['id']
                            st.rerun()

    elif st.session_state.page == "Skrining AI & OLD CARTS":
        page_screening(user, model)
    elif st.session_state.page == "Rekam Medis":
        page_records(user)

main()
