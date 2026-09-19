"""
MAMMOUTH — MY ASSISTANT IN MOUTH HEALTH
=========================================================
Versi   : 7.0  "Rhythm Clinical"
Stack   : Streamlit + SQLite + (opsional) Ultralytics YOLO + (opsional) fpdf2
Fitur   : Multi-User Auth (salted PBKDF2), Manajemen Pasien, SQLite Database,
          Data Isolation per klinisi, OLD CARTS Anamnesis, Batch Upload,
          Live Preview, Crop & Image Enhancement, AI + Manual Hybrid Detection,
          Sintesis Diagnosis Klinis + Diagnosis Banding, Penyimpanan Citra (BLOB),
          Riwayat EMR (cari/filter/status/catatan/hapus) dgn thumbnail,
          Ekspor CSV / Excel / PDF (dgn citra terlampir), Dashboard Analitik
          (tren, donat urgensi, statistik), Ensiklopedia Lesi, Panel Admin,
          Bilah Atas (pencarian cepat), Navigasi Cepat Kasus Darurat,
          Pengaturan Profil & Preferensi, Backup Basis Data.
=========================================================
"""


import io
import os
import uuid
import base64
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path


import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw, ImageFont


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False


# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
APP_VERSION = "7.0 Rhythm Clinical"
APP_NAME = "MAMMOUTH"
APP_TAGLINE = "My Assistant In Mouth Health"
DB_FILE = "mammouth.db"


# ------------------------------------------------------------
# DESIGN TOKENS — palet "Rhythm Clinical": teal admin-dashboard + aksen multi-warna
# (nama variabel dipertahankan dari versi sebelumnya agar semua rujukan CSS tetap valid;
#  nilai warnanya diperbarui mengikuti referensi desain baru)
# ------------------------------------------------------------
NAVY_900 = "#101828"      # Judul & teks utama (slate gelap)
NAVY_700 = "#0EA6A2"      # Warna merek utama (teal — brand primary)
NAVY_600 = "#14B8B0"      # Gradasi / hover teal
NAVY_100 = "#E1FAF6"      # Latar lembut teal (nav aktif, pill)
CORAL_500 = "#F43F5E"     # Aksen darurat / CTA hangat (rose)
CORAL_600 = "#E11D48"     # Hover aksen rose
CORAL_100 = "#FFE4E8"     # Latar lembut rose
SLATE_500 = "#64748B"     # Teks sekunder
SLATE_300 = "#CBD5E1"
BORDER_COLOR = "#EAEEF4"
BG_MIST = "#F2F4F9"       # Latar halaman
WHITE = "#FFFFFF"


DANGER = "#EF4444"
DANGER_BG = "#FEF2F2"
WARNING = "#F59E0B"
WARNING_BG = "#FFFBEB"
SUCCESS = "#16A34A"
SUCCESS_BG = "#F0FDF4"


# Aksen tambahan khas dasbor admin (variasi warna kartu KPI & badge)
CYAN_500 = "#22B8E0"      # Aksen sekunder (appointments/notifikasi)
INDIGO_500 = "#4F46E5"    # Aksen sorot (kartu spotlight)
INDIGO_600 = "#4338CA"
INDIGO_100 = "#EAE9FD"
AMBER_500 = WARNING       # Aksen kuning/oranye (encounters)
DARK_PILL = "#111827"     # Pill gelap (mis. "Referral"/kategori arsip)


# Dipertahankan agar kompatibel dengan nama variabel versi sebelumnya
PRIMARY = NAVY_700
PRIMARY_LIGHT = NAVY_100
ACCENT = CORAL_500
DARK_TEXT = NAVY_900
GRAY_TEXT = SLATE_500


STATUS_OPTIONS = ["Baru", "Perlu Tindak Lanjut", "Selesai"]


LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum (Cheek Biting)",
        "deskripsi": "Kebiasaan menggigit mukosa bukal, mengakibatkan tampilan kasar/bergerigi. Dapat berpotensi menyebabkan ulserasi. Sering terkait faktor stres psikologis.",
        "rekomendasi": "Edukasi penghentian kebiasaan buruk (habit breaking); evaluasi ulang bila lesi menetap >2 minggu.",
        "urgensi": "Rendah",
        "kategori": "Kebiasaan & Trauma",
        "diagnosis_banding": ["Leukoplakia friksional", "White sponge nevus", "Kandidiasis pseudomembran"],
        "ikon": "🦷",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue",
        "deskripsi": "Permukaan lidah tertutup selaput pseudomembran akibat penumpukan debris, keratin tidak terdeskuamasi, dan mikroorganisme.",
        "rekomendasi": "Instruksikan pembersihan mekanis rutin (tongue scraper) dan evaluasi oral hygiene.",
        "urgensi": "Rendah",
        "kategori": "Kebersihan Mulut",
        "diagnosis_banding": ["Kandidiasis oral (thrush)", "Hairy tongue (Lingua villosa nigra)", "Leukoplakia"],
        "ikon": "👅",
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Demineralisasi jaringan keras gigi oleh asam hasil metabolisme bakteri plak.",
        "rekomendasi": "Pemeriksaan klinis (sondasi/perkusi) dan radiografis lanjutan untuk rencana restorasi atau perawatan saluran akar.",
        "urgensi": "Sedang-Tinggi",
        "kategori": "Restoratif",
        "diagnosis_banding": ["Erosi gigi", "Abrasi/atrisi", "Hipoplasia enamel developmental"],
        "ikon": "🦠",
    },
    "linea alba": {
        "nama_klinis": "Linea Alba Buccalis",
        "deskripsi": "Garis putih horizontal pada mukosa bukal setinggi bidang oklusal, umumnya akibat tekanan atau friksi oklusal ringan.",
        "rekomendasi": "Bersifat jinak dan fisiologis, umumnya tidak memerlukan tatalaksana khusus.",
        "urgensi": "Rendah",
        "kategori": "Variasi Anatomis",
        "diagnosis_banding": ["Leukoplakia friksional", "White sponge nevus"],
        "ikon": "➖",
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran vena (varises) pada permukaan ventral lidah, temuan umum pada individu usia lanjut.",
        "rekomendasi": "Tidak memerlukan tindakan invasif. Edukasi pasien terkait sifat jinak lesi.",
        "urgensi": "Rendah",
        "kategori": "Variasi Anatomis",
        "diagnosis_banding": ["Hemangioma", "Varix vena", "Trombosis vena superfisial"],
        "ikon": "🔵",
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Deposit terkalsifikasi (kalkulus) dan diskolorasi ekstrinsik pada permukaan gigi.",
        "rekomendasi": "Tindakan scaling dan root planing (SRP) profesional; instruksi DHE.",
        "urgensi": "Sedang",
        "kategori": "Kebersihan Mulut",
        "diagnosis_banding": ["Stain ekstrinsik (tembakau/kopi/teh)", "Diskolorasi intrinsik gigi"],
        "ikon": "✨",
    },
    "torus": {
        "nama_klinis": "Torus (Palatinus/Mandibularis)",
        "deskripsi": "Eksostosis tulang jinak, umumnya asimtomatik dan lambat membesar.",
        "rekomendasi": "Observasi. Pembedahan hanya diindikasikan bila mengganggu fungsi bicara/pengunyahan atau sebagai persiapan protesa.",
        "urgensi": "Rendah",
        "kategori": "Variasi Anatomis",
        "diagnosis_banding": ["Osteoma", "Eksostosis reaktif", "Odontoma (jarang)"],
        "ikon": "⬜",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Lesi ulseratif mukosa oral sekunder akibat trauma mekanis (tergigit/gesekan), termal, atau kimiawi.",
        "rekomendasi": "Eliminasi faktor kausatif. Evaluasi ulang dalam 10-14 hari untuk menyingkirkan diagnosis banding keganasan.",
        "urgensi": "Sedang",
        "kategori": "Kebiasaan & Trauma",
        "diagnosis_banding": ["Stomatitis aftosa rekuren (SAR)", "Lesi herpetik", "Luka bakar kimiawi", "Karsinoma sel skuamosa (jika tidak sembuh >2 minggu — rujuk segera)"],
        "ikon": "⭕",
    },
}
LESION_KEYS = list(LESION_INFO.keys())


st.set_page_config(
    page_title=f"{APP_NAME} · AI Oral Health",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# INISIALISASI SESSION STATE
# ============================================================
_DEFAULT_STATE = {
    "logged_in": False,
    "active_user": None,
    "user_name": None,
    "user_role": None,
    "is_admin": False,
    "anamnesis_data": None,
    "yolo_version": "YOLOv8",
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "active_patient": None,
    "detection_results": None,
    "detection_cache": {},
    "confirm_delete_log": None,
    "confirm_delete_patient": None,
    "confirm_wipe": False,
    "theme_mode": "Terang",
}
for _k, _v in _DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# LAPISAN DATABASE (SQLITE)
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn




def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT,
                    salt TEXT,
                    full_name TEXT,
                    role TEXT,
                    is_admin INTEGER DEFAULT 0,
                    pref_conf REAL DEFAULT 0.25,
                    pref_iou REAL DEFAULT 0.45,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    owner_user TEXT,
                    name TEXT,
                    age TEXT,
                    gender TEXT,
                    phone TEXT,
                    notes TEXT,
                    created_at TEXT,
                    FOREIGN KEY(owner_user) REFERENCES users(username)
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS emr_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    patient_id TEXT,
                    patient_name TEXT,
                    waktu TEXT,
                    tanggal TEXT,
                    lesi_terdeteksi TEXT,
                    confidence REAL,
                    model_version TEXT,
                    nama_file TEXT,
                    o_onset TEXT, l_location TEXT, d_duration TEXT, c_character TEXT,
                    a_aggravating TEXT, r_relieving TEXT, t_timing TEXT, s_severity INTEGER,
                    suspek_diagnosis TEXT,
                    diagnosis_banding TEXT,
                    status TEXT DEFAULT 'Baru',
                    catatan_tambahan TEXT,
                    image_blob BLOB,
                    FOREIGN KEY(user_id) REFERENCES users(username)
                )""")
    conn.commit()
    conn.close()




def migrate_db():
    """Migrasi ringan: menambah kolom baru pada basis data versi lama tanpa menghapus data."""
    conn = get_conn()
    c = conn.cursor()
    for stmt in (
        "ALTER TABLE emr_logs ADD COLUMN diagnosis_banding TEXT",
        "ALTER TABLE emr_logs ADD COLUMN image_blob BLOB",
    ):
        try:
            c.execute(stmt)
        except sqlite3.OperationalError:
            pass  # kolom sudah ada
    conn.commit()
    conn.close()




init_db()
migrate_db()


# ------------------------------------------------------------
# AUTENTIKASI
# ------------------------------------------------------------
def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = uuid.uuid4().hex
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return pwd_hash, salt




def create_user(username, password, full_name, role):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    is_first_user = c.fetchone()[0] == 0
    pwd_hash, salt = hash_password(password)
    try:
        c.execute("""INSERT INTO users (username, password_hash, salt, full_name, role, is_admin, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                  (username.lower().strip(), pwd_hash, salt, full_name.strip(), role.strip(),
                   1 if is_first_user else 0, datetime.now().isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()




def verify_login(username, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username.lower().strip(),))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    check_hash, _ = hash_password(password, row["salt"])
    if check_hash == row["password_hash"]:
        return dict(row)
    return None




def update_password(username, new_password):
    pwd_hash, salt = hash_password(new_password)
    conn = get_conn()
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?", (pwd_hash, salt, username))
    conn.commit()
    conn.close()




def update_profile(username, full_name, role):
    conn = get_conn()
    conn.execute("UPDATE users SET full_name=?, role=? WHERE username=?", (full_name, role, username))
    conn.commit()
    conn.close()




def save_prefs(username, conf, iou):
    conn = get_conn()
    conn.execute("UPDATE users SET pref_conf=?, pref_iou=? WHERE username=?", (conf, iou, username))
    conn.commit()
    conn.close()




def count_users():
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n




# ------------------------------------------------------------
# MANAJEMEN PASIEN
# ------------------------------------------------------------
def create_patient(owner, name, age, gender, phone, notes):
    pid = uuid.uuid4().hex[:10]
    conn = get_conn()
    conn.execute("""INSERT INTO patients (patient_id, owner_user, name, age, gender, phone, notes, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                 (pid, owner, name.strip(), str(age), gender, phone, notes, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return pid




def get_patients(owner) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM patients WHERE owner_user=? ORDER BY name ASC", conn, params=(owner,))
    conn.close()
    return df




def get_patient(patient_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None




def update_patient(patient_id, name, age, gender, phone, notes):
    conn = get_conn()
    conn.execute("""UPDATE patients SET name=?, age=?, gender=?, phone=?, notes=? WHERE patient_id=?""",
                 (name, str(age), gender, phone, notes, patient_id))
    conn.commit()
    conn.close()




def delete_patient(patient_id, owner):
    conn = get_conn()
    conn.execute("DELETE FROM patients WHERE patient_id=? AND owner_user=?", (patient_id, owner))
    conn.commit()
    conn.close()




# ------------------------------------------------------------
# EMR / RIWAYAT PEMERIKSAAN
# ------------------------------------------------------------
def append_logs(records: list):
    if not records:
        return
    conn = get_conn()
    c = conn.cursor()
    for r in records:
        c.execute("""INSERT INTO emr_logs
                     (log_id, user_id, patient_id, patient_name, waktu, tanggal, lesi_terdeteksi, confidence,
                      model_version, nama_file, o_onset, l_location, d_duration, c_character, a_aggravating,
                      r_relieving, t_timing, s_severity, suspek_diagnosis, diagnosis_banding, status,
                      catatan_tambahan, image_blob)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (r["ID"], r["user_id"], r.get("patient_id"), r.get("patient_name", "Umum"),
                   r["Waktu"], r["Tanggal"], r["Lesi_Terdeteksi"], r["Confidence"], r["Model_Version"],
                   r["Nama_File"], r.get("O_Onset", "-"), r.get("L_Location", "-"), r.get("D_Duration", "-"),
                   r.get("C_Character", "-"), r.get("A_Aggravating", "-"), r.get("R_Relieving", "-"),
                   r.get("T_Timing", "-"), r.get("S_Severity", 0), r["Suspek_Diagnosis"],
                   r.get("Diagnosis_Banding", "-"), r.get("Status", "Baru"), r.get("Catatan", ""),
                   r.get("Image_Blob")))
    conn.commit()
    conn.close()




def load_user_logs(username) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM emr_logs WHERE user_id=? ORDER BY tanggal DESC, waktu DESC",
                      conn, params=(username,))
    conn.close()
    return df




def update_log(log_id, status=None, catatan=None):
    conn = get_conn()
    if status is not None:
        conn.execute("UPDATE emr_logs SET status=? WHERE log_id=?", (status, log_id))
    if catatan is not None:
        conn.execute("UPDATE emr_logs SET catatan_tambahan=? WHERE log_id=?", (catatan, log_id))
    conn.commit()
    conn.close()




def delete_log(log_id, owner):
    conn = get_conn()
    conn.execute("DELETE FROM emr_logs WHERE log_id=? AND user_id=?", (log_id, owner))
    conn.commit()
    conn.close()




def load_all_logs_admin() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("""SELECT emr_logs.*, users.full_name AS clinician_name
                         FROM emr_logs LEFT JOIN users ON emr_logs.user_id = users.username
                         ORDER BY tanggal DESC, waktu DESC""", conn)
    conn.close()
    return df




def load_all_users_admin() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT username, full_name, role, is_admin, created_at FROM users ORDER BY created_at ASC", conn)
    conn.close()
    return df




# ============================================================
# EKSPOR DATA (CSV / EXCEL / PDF)
# ============================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")




def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Riwayat EMR")
        return buf.getvalue()
    except Exception:
        return b""




def build_pdf_report(record: dict, clinician_name: str) -> bytes:
    """Membuat laporan pemeriksaan 1 halaman dalam format PDF. Mengembalikan b'' jika fpdf2 tidak tersedia."""
    if not FPDF_OK:
        return b""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 30, 61)
    pdf.cell(0, 10, f"{APP_NAME} — Laporan Skrining Oral", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Dibuat oleh {clinician_name} pada {record.get('Tanggal','-')} {record.get('Waktu','-')}", ln=True)
    pdf.ln(4)


    def row(label, value):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 30, 61)
        pdf.cell(45, 7, str(label), border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 7, str(value if value not in (None, "") else "-"))


    pdf.set_draw_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(242, 112, 60)
    pdf.cell(0, 8, "Data Pasien", ln=True)
    row("Nama Pasien", record.get("patient_name", "Umum"))
    pdf.ln(2)


    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(242, 112, 60)
    pdf.cell(0, 8, "Anamnesis (OLD CARTS)", ln=True)
    row("Onset", record.get("O_Onset"))
    row("Lokasi", record.get("L_Location"))
    row("Durasi", record.get("D_Duration"))
    row("Karakter", record.get("C_Character"))
    row("Faktor Memperberat", record.get("A_Aggravating"))
    row("Faktor Meredakan", record.get("R_Relieving"))
    row("Waktu Muncul", record.get("T_Timing"))
    row("Skala Nyeri (VAS)", f"{record.get('S_Severity', 0)}/10")
    pdf.ln(2)


    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(242, 112, 60)
    pdf.cell(0, 8, "Temuan AI & Sintesis Klinis", ln=True)
    row("Lesi Terdeteksi", record.get("Lesi_Terdeteksi"))
    row("Model", record.get("Model
