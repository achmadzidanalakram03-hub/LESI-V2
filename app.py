"""
==================================================================================
 CDSS KEDOKTERAN GIGI (Clinical Decision Support System)
==================================================================================
Aplikasi berbasis Streamlit untuk membantu dokter gigi & koas dalam:
    1. Melakukan anamnesis terstruktur
    2. Mendeteksi lesi rongga mulut secara visual menggunakan model YOLO (best.pt)
    3. Menyusun draf rencana perawatan
    4. Menyimpan & menelusuri rekam medis digital (SQLite)

Struktur kode (dipisah per fungsi/tanggung jawab):
    - BAGIAN 1  : Konfigurasi, import, tema CSS
    - BAGIAN 2  : Fungsi-fungsi Database (SQLite)
    - BAGIAN 3  : Fungsi-fungsi Inference YOLO (ultralytics)
    - BAGIAN 4  : Fungsi bantu (utilitas gambar, draf rencana perawatan)
    - BAGIAN 5  : Halaman-halaman UI (Beranda, Deteksi & Anamnesis, Riwayat)
    - BAGIAN 6  : Main App & Navigasi Sidebar

CATATAN PENTING:
    - Sesuaikan dictionary TREATMENT_SUGGESTIONS (BAGIAN 4) dengan nama-nama
      kelas (class names) hasil training model 'best.pt' Anda sendiri.
    - Aplikasi ini adalah ALAT BANTU (decision support), bukan pengganti
      diagnosis & penilaian klinis dokter gigi.
==================================================================================
"""

# ==================================================================================
# BAGIAN 1: KONFIGURASI & IMPORT
# ==================================================================================
import os
import base64
from io import BytesIO
from datetime import datetime, date, timedelta

import sqlite3
import pandas as pd
import streamlit as st
from PIL import Image

DB_NAME = "rekam_medis.db"
MODEL_PATH = "best.pt"

st.set_page_config(
    page_title="CDSS Kedokteran Gigi",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Memuat tema visual medis (biru & putih) untuk seluruh aplikasi."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }
        .main { background-color: #f4f8fb; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d47a1 0%, #1565c0 100%);
        }
        section[data-testid="stSidebar"] * { color: #ffffff !important; }
        section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25); }

        /* Header banner */
        .cdss-header {
            background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
            padding: 1.6rem 2rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 14px rgba(13,71,161,0.25);
        }
        .cdss-header h1 { margin: 0; font-size: 1.9rem; }
        .cdss-header p { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

        /* Card container */
        .cdss-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #e3edf7;
            margin-bottom: 1rem;
        }

        /* Metric styling */
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e3edf7;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        /* Buttons */
        .stButton>button {
            background-color: #1565c0;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: 0.15s;
        }
        .stButton>button:hover { background-color: #0d47a1; color: white; }

        /* Disclaimer box */
        .cdss-disclaimer {
            background: #fff8e1;
            border-left: 5px solid #f9a825;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        /* Section title */
        .cdss-section-title {
            color: #0d47a1;
            font-weight: 700;
            font-size: 1.05rem;
            border-bottom: 2px solid #e3edf7;
            padding-bottom: 0.3rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# BAGIAN 2: FUNGSI-FUNGSI DATABASE (SQLite)
# ==================================================================================
def get_connection():
    """Membuka koneksi baru ke database SQLite."""
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    """Inisialisasi tabel rekam_medis jika belum ada."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rekam_medis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal_waktu TEXT NOT NULL,
            nama_pasien TEXT NOT NULL,
            usia INTEGER,
            keluhan_utama TEXT,
            skala_nyeri INTEGER,
            durasi_keluhan TEXT,
            riwayat_sistemik TEXT,
            hasil_deteksi TEXT,
            rencana_perawatan TEXT,
            gambar_base64 TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_record(data: dict):
    """Menyimpan satu rekam medis baru ke database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rekam_medis
        (tanggal_waktu, nama_pasien, usia, keluhan_utama, skala_nyeri,
         durasi_keluhan, riwayat_sistemik, hasil_deteksi, rencana_perawatan, gambar_base64)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            data["tanggal_waktu"], data["nama_pasien"], data["usia"],
            data["keluhan_utama"], data["skala_nyeri"], data["durasi_keluhan"],
            data["riwayat_sistemik"], data["hasil_deteksi"],
            data["rencana_perawatan"], data["gambar_base64"],
        ),
    )
    conn.commit()
    conn.close()


def get_all_records() -> pd.DataFrame:
    """Mengambil seluruh rekam medis. Mengembalikan DataFrame kosong bila tabel kosong/error."""
    try:
        conn = get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM rekam_medis ORDER BY tanggal_waktu DESC", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_metrics():
    """Menghitung jumlah rekam medis hari ini & bulan ini untuk dashboard."""
    df = get_all_records()
    if df.empty:
        return 0, 0
    df["tgl_dt"] = pd.to_datetime(df["tanggal_waktu"], errors="coerce")
    today = date.today()
    total_today = int((df["tgl_dt"].dt.date == today).sum())
    total_month = int(
        ((df["tgl_dt"].dt.month == today.month) & (df["tgl_dt"].dt.year == today.year)).sum()
    )
    return total_today, total_month


# ==================================================================================
# BAGIAN 3: FUNGSI-FUNGSI INFERENCE YOLO (ultralytics)
# ==================================================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    """
    Memuat model YOLO dari file lokal. Aman terhadap error:
    - Jika library ultralytics belum terinstal
    - Jika file model belum ditemukan
    Mengembalikan tuple (model, error_message). error_message None jika sukses.
    """
    if not os.path.exists(model_path):
        return None, f"File model '{model_path}' tidak ditemukan di direktori aplikasi."
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, f"Gagal memuat model YOLO: {e}"


def run_inference(model, image: Image.Image):
    """
    Menjalankan deteksi objek pada gambar menggunakan model YOLO.
    Mengembalikan (annotated_image: PIL.Image, detections: list[dict], error: str|None)
    detections berisi list of {"label": str, "confidence": float}
    """
    try:
        results = model.predict(image, verbose=False)
        result = results[0]

        # result.plot() -> numpy array BGR dengan bounding box, label, confidence tergambar
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[..., ::-1]
        annotated_image = Image.fromarray(annotated_rgb)

        names = model.names  # dict {id: nama_kelas} atau list
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if isinstance(names, dict):
                    label = names.get(cls_id, str(cls_id))
                else:
                    label = names[cls_id] if cls_id < len(names) else str(cls_id)
                detections.append({"label": label, "confidence": conf})

        return annotated_image, detections, None
    except Exception as e:
        return None, [], f"Terjadi kesalahan saat proses inference: {e}"


# ==================================================================================
# BAGIAN 4: FUNGSI BANTU (utilitas gambar & draf rencana perawatan)
# ==================================================================================
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))


# Sesuaikan key dictionary ini dengan nama kelas hasil training model best.pt Anda.
TREATMENT_SUGGESTIONS = {
    "karies": (
        "- Evaluasi kedalaman karies (superfisial/media/profunda) melalui pemeriksaan "
        "klinis & radiografis\n"
        "- Pertimbangkan restorasi langsung (tumpatan) sesuai indikasi\n"
        "- Bila karies profunda: pertimbangkan pulp capping / perawatan saluran akar\n"
        "- Edukasi kontrol plak, diet rendah gula, dan menyikat gigi 2x sehari"
    ),
    "calculus": (
        "- Rencanakan scaling (pembersihan karang gigi)\n"
        "- Edukasi teknik menyikat gigi dan flossing yang benar\n"
        "- Kontrol rutin setiap 6 bulan"
    ),
    "gingivitis": (
        "- Scaling & root planing bila diperlukan\n"
        "- Edukasi kebersihan mulut (oral hygiene instruction)\n"
        "- Evaluasi ulang setelah 2-4 minggu"
    ),
    "periodontitis": (
        "- Pertimbangkan rujukan ke spesialis periodonsia bila derajat sedang-berat\n"
        "- Scaling root planing bertahap\n"
        "- Evaluasi kegoyangan gigi dan kedalaman poket periodontal"
    ),
    "ulcer": (
        "- Identifikasi kemungkinan penyebab (trauma, aftosa, infeksi)\n"
        "- Pemberian obat topikal sesuai indikasi\n"
        "- Evaluasi ulang bila tidak sembuh dalam 2 minggu; rujuk bila curiga keganasan"
    ),
}


def generate_treatment_draft(detections: list) -> str:
    """Membuat draf rencana perawatan otomatis berdasarkan label yang terdeteksi."""
    if not detections:
        return (
            "Tidak ada temuan lesi spesifik dari deteksi visual AI.\n"
            "Rencana perawatan dapat disusun berdasarkan hasil anamnesis dan "
            "pemeriksaan klinis langsung oleh dokter gigi."
        )

    lines, seen = [], set()
    for d in detections:
        label_raw = d["label"]
        label_key = label_raw.lower().replace(" ", "_")
        if label_key in seen:
            continue
        seen.add(label_key)
        lines.append(f"🔹 Temuan: {label_raw.title()} (confidence: {d['confidence']*100:.1f}%)")
        lines.append(
            TREATMENT_SUGGESTIONS.get(
                label_key,
                "- Perlu evaluasi klinis lebih lanjut oleh dokter gigi\n"
                "- Pertimbangkan pemeriksaan penunjang (radiograf) bila diperlukan",
            )
        )
        lines.append("")

    lines.append(
        "⚠️ Draf ini adalah saran awal berbasis temuan AI. "
        "Harap sesuaikan dengan pemeriksaan klinis langsung."
    )
    return "\n".join(lines)


# ==================================================================================
# BAGIAN 5: HALAMAN-HALAMAN UI
# ==================================================================================
def page_beranda():
    st.markdown(
        """
        <div class="cdss-header">
            <h1>🦷 CDSS Kedokteran Gigi</h1>
            <p>Sistem Pendukung Keputusan Klinis untuk membantu diagnosis, edukasi pasien,
            dan perencanaan perawatan gigi berbasis Anamnesis &amp; Deteksi Visual (AI/YOLO)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_today, total_month = get_metrics()
    df_all = get_all_records()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🗓️ Pasien Hari Ini", total_today)
    with col2:
        st.metric("📅 Pasien Bulan Ini", total_month)
    with col3:
        st.metric("🗂️ Total Rekam Medis", len(df_all))

    st.write("")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("#### 📌 Tentang Aplikasi")
        st.write(
            """
Aplikasi ini dirancang untuk membantu dokter gigi dan koas dalam:
- Mempertimbangkan kemungkinan diagnosis berdasarkan hasil **anamnesis** dan
  **deteksi visual lesi** menggunakan model AI (YOLO).
- Menyusun draf **rencana perawatan** awal.
- Mendokumentasikan dan menyimpan **rekam medis digital** pasien secara terstruktur.
            """
        )
        st.info("➡️ Gunakan menu **Deteksi & Anamnesis** di sidebar untuk memulai pemeriksaan pasien baru.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Status Model AI")
        if os.path.exists(MODEL_PATH):
            st.success(f"Model `{MODEL_PATH}` ditemukan dan siap digunakan.")
        else:
            st.warning(f"File `{MODEL_PATH}` belum ditemukan di direktori aplikasi.")
        st.caption(f"Total baris database: {len(df_all)}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="cdss-disclaimer">
        ⚠️ <b>Disclaimer:</b> Aplikasi ini merupakan alat bantu pengambilan keputusan klinis
        (Clinical Decision Support System) dan <b>tidak menggantikan</b> pemeriksaan klinis
        serta penilaian profesional dokter gigi.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_deteksi():
    st.markdown(
        """
        <div class="cdss-header">
            <h1>🔬 Deteksi &amp; Anamnesis</h1>
            <p>Input data anamnesis pasien dan unggah gambar klinis untuk dianalisis oleh model AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cdss-disclaimer">⚠️ Hasil deteksi AI bersifat sebagai '
        '<b>pendukung keputusan</b>, bukan diagnosis final. Selalu lakukan konfirmasi klinis.</div>',
        unsafe_allow_html=True,
    )

    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    if "deteksi" not in st.session_state:
        st.session_state.deteksi = None

    model, model_error = load_yolo_model(MODEL_PATH)

    col_input, col_hasil = st.columns([1, 1.2], gap="large")

    # ---------------- KOLOM KIRI: INPUT ----------------
    with col_input:
        st.markdown('<p class="cdss-section-title">📝 Form Anamnesis</p>', unsafe_allow_html=True)

        form_key = f"form_anamnesis_{st.session_state.reset_counter}"
        uploader_key = f"uploader_{st.session_state.reset_counter}"

        with st.form(key=form_key):
            nama_pasien = st.text_input("Nama Pasien / ID Pasien *")

            c1, c2 = st.columns(2)
            with c1:
                usia = st.number_input("Usia (tahun)", min_value=0, max_value=120, step=1)
            with c2:
                skala_nyeri = st.slider("Skala Nyeri (0-10)", 0, 10, 0)

            keluhan_utama = st.text_area("Keluhan Utama *", height=90)

            c3, c4 = st.columns(2)
            with c3:
                durasi_keluhan = st.text_input("Durasi Keluhan", placeholder="cth: 3 hari")
            with c4:
                riwayat_sistemik = st.text_input(
                    "Riwayat Sistemik", placeholder="cth: Diabetes, Hipertensi, tidak ada"
                )

            st.markdown("---")
            st.markdown("**📷 Upload Gambar Klinis**")
            uploaded_file = st.file_uploader(
                "Format: JPG / PNG", type=["jpg", "jpeg", "png"], key=uploader_key
            )

            submitted = st.form_submit_button("🔍 Analisis", use_container_width=True)

        if st.button("🔄 Formulir Baru", use_container_width=True):
            st.session_state.reset_counter += 1
            st.session_state.deteksi = None
            st.rerun()

        if submitted:
            if not nama_pasien.strip():
                st.error("Nama Pasien / ID wajib diisi.")
            elif not keluhan_utama.strip():
                st.error("Keluhan Utama wajib diisi.")
            elif uploaded_file is None:
                st.error("Mohon unggah gambar klinis terlebih dahulu.")
            elif model is None:
                st.error(f"Model YOLO tidak dapat digunakan: {model_error}")
            else:
                with st.spinner("Menganalisis gambar dengan model AI..."):
                    try:
                        image = Image.open(uploaded_file).convert("RGB")
                        annotated_image, detections, infer_error = run_inference(model, image)
                    except Exception as e:
                        annotated_image, detections, infer_error = None, [], str(e)

                if infer_error:
                    st.error(infer_error)
                else:
                    st.session_state.deteksi = {
                        "nama_pasien": nama_pasien.strip(),
                        "usia": int(usia),
                        "keluhan_utama": keluhan_utama.strip(),
                        "skala_nyeri": int(skala_nyeri),
                        "durasi_keluhan": durasi_keluhan.strip(),
                        "riwayat_sistemik": riwayat_sistemik.strip(),
                        "annotated_image": annotated_image,
                        "detections": detections,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.success("✅ Analisis selesai! Lihat hasil di panel kanan.")

    # ---------------- KOLOM KANAN: HASIL & EDUKASI ----------------
    with col_hasil:
        st.markdown('<p class="cdss-section-title">📊 Hasil &amp; Edukasi</p>', unsafe_allow_html=True)

        data = st.session_state.deteksi
        if not data:
            st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
            st.write(
                "Belum ada hasil analisis. Silakan lengkapi form anamnesis dan "
                "unggah gambar, lalu klik **Analisis**."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        detections = data["detections"]

        # Hasil visual & metrik temuan
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("##### 🖼️ Hasil Visual Deteksi")
        if data["annotated_image"] is not None:
            st.image(
                data["annotated_image"],
                use_container_width=True,
                caption="Hasil Bounding Box Deteksi YOLO",
            )
        else:
            st.warning("Gambar hasil deteksi tidak tersedia.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Jumlah Temuan", len(detections))
        with m2:
            avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) if detections else 0
            st.metric("Rata-rata Confidence", f"{avg_conf*100:.1f}%")
        with m3:
            st.metric("Jenis Lesi Terdeteksi", len(set(d["label"] for d in detections)))

        if detections:
            df_det = pd.DataFrame(detections)
            df_det["confidence"] = (df_det["confidence"] * 100).round(2).astype(str) + "%"
            df_det.columns = ["Label / Kelas", "Confidence"]
            st.dataframe(df_det, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada objek/lesi yang terdeteksi pada gambar.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Ringkasan anamnesis
        with st.expander("📋 Ringkasan Data Anamnesis"):
            st.write(f"**Nama Pasien/ID:** {data['nama_pasien']}")
            st.write(f"**Usia:** {data['usia']} tahun")
            st.write(f"**Keluhan Utama:** {data['keluhan_utama']}")
            st.write(f"**Skala Nyeri:** {data['skala_nyeri']}/10")
            st.write(f"**Durasi Keluhan:** {data['durasi_keluhan'] or '-'}")
            st.write(f"**Riwayat Sistemik:** {data['riwayat_sistemik'] or '-'}")

        # Draf rencana perawatan (editable)
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("##### 🩺 Draf Rencana Perawatan")
        rencana_perawatan = st.text_area(
            "Rencana perawatan dapat diedit sesuai penilaian klinis dokter",
            value=generate_treatment_draft(detections),
            height=200,
            key=f"rencana_{data['timestamp']}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Simpan rekam medis
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("##### 💾 Simpan Rekam Medis")
        if st.button("💾 Simpan Rekam Medis", type="primary", use_container_width=True):
            try:
                img_b64 = (
                    image_to_base64(data["annotated_image"])
                    if data["annotated_image"] is not None
                    else ""
                )
                hasil_deteksi_str = (
                    "; ".join(f"{d['label']} ({d['confidence']*100:.1f}%)" for d in detections)
                    if detections
                    else "Tidak ada temuan"
                )
                insert_record(
                    {
                        "tanggal_waktu": data["timestamp"],
                        "nama_pasien": data["nama_pasien"],
                        "usia": data["usia"],
                        "keluhan_utama": data["keluhan_utama"],
                        "skala_nyeri": data["skala_nyeri"],
                        "durasi_keluhan": data["durasi_keluhan"],
                        "riwayat_sistemik": data["riwayat_sistemik"],
                        "hasil_deteksi": hasil_deteksi_str,
                        "rencana_perawatan": rencana_perawatan,
                        "gambar_base64": img_b64,
                    }
                )
                st.success(f"✅ Rekam medis untuk **{data['nama_pasien']}** berhasil disimpan!")
                st.balloons()
                st.session_state.reset_counter += 1
                st.session_state.deteksi = None
            except Exception as e:
                st.error(f"Gagal menyimpan rekam medis: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


def page_riwayat():
    st.markdown(
        """
        <div class="cdss-header">
            <h1>📋 Riwayat Pasien &amp; Rekam Medis</h1>
            <p>Lihat, filter, dan telusuri rekam medis pasien yang telah tersimpan</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = get_all_records()
    if df.empty:
        st.info(
            "Belum ada data rekam medis yang tersimpan. Silakan lakukan analisis pasien "
            "terlebih dahulu di menu **Deteksi & Anamnesis**."
        )
        return

    df["tgl_dt"] = pd.to_datetime(df["tanggal_waktu"], errors="coerce")

    # ---------------- FILTER ----------------
    st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
    st.markdown("##### 🔎 Filter Data")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        filter_mode = st.selectbox(
            "Rentang Waktu", ["Semua Data", "Hari Ini", "Bulan Ini", "Pilih Rentang Tanggal"]
        )
    start_date, end_date = None, None
    if filter_mode == "Pilih Rentang Tanggal":
        with col_f2:
            start_date = st.date_input("Dari Tanggal", value=date.today() - timedelta(days=7))
        with col_f3:
            end_date = st.date_input("Sampai Tanggal", value=date.today())
    st.markdown("</div>", unsafe_allow_html=True)

    today = date.today()
    if filter_mode == "Semua Data":
        df_filtered = df
    elif filter_mode == "Hari Ini":
        df_filtered = df[df["tgl_dt"].dt.date == today]
    elif filter_mode == "Bulan Ini":
        df_filtered = df[(df["tgl_dt"].dt.month == today.month) & (df["tgl_dt"].dt.year == today.year)]
    else:
        if start_date and end_date:
            df_filtered = df[(df["tgl_dt"].dt.date >= start_date) & (df["tgl_dt"].dt.date <= end_date)]
        else:
            df_filtered = df

    # ---------------- TABEL DAFTAR PASIEN ----------------
    st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
    st.markdown(f"##### 📑 Daftar Pasien ({len(df_filtered)} rekam medis)")
    if df_filtered.empty:
        st.warning("Tidak ada data pada rentang waktu yang dipilih.")
    else:
        tabel = df_filtered[["id", "tanggal_waktu", "nama_pasien", "keluhan_utama", "hasil_deteksi"]].copy()
        tabel.columns = ["ID", "Tanggal/Waktu", "Nama Pasien", "Keluhan Utama", "Hasil Deteksi"]
        st.dataframe(tabel, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DETAIL REKAM MEDIS ----------------
    if not df_filtered.empty:
        st.markdown('<div class="cdss-card">', unsafe_allow_html=True)
        st.markdown("##### 🗂️ Detail Rekam Medis")

        options = {
            f"{row['nama_pasien']} — {row['tanggal_waktu']} (ID:{row['id']})": row["id"]
            for _, row in df_filtered.iterrows()
        }
        selected_label = st.selectbox("Pilih Pasien", list(options.keys()))
        selected_id = options[selected_label]
        record = df_filtered[df_filtered["id"] == selected_id].iloc[0]

        col_d1, col_d2 = st.columns([1, 1.2])
        with col_d1:
            st.markdown("**Data Anamnesis**")
            st.write(f"- **Nama Pasien/ID:** {record['nama_pasien']}")
            st.write(f"- **Usia:** {record['usia']} tahun")
            st.write(f"- **Tanggal/Waktu:** {record['tanggal_waktu']}")
            st.write(f"- **Keluhan Utama:** {record['keluhan_utama']}")
            st.write(f"- **Skala Nyeri:** {record['skala_nyeri']}/10")
            st.write(f"- **Durasi Keluhan:** {record['durasi_keluhan'] or '-'}")
            st.write(f"- **Riwayat Sistemik:** {record['riwayat_sistemik'] or '-'}")
            st.write(f"- **Hasil Deteksi:** {record['hasil_deteksi']}")

            with st.expander("🩺 Rencana Perawatan", expanded=True):
                st.write(record["rencana_perawatan"] or "-")

        with col_d2:
            st.markdown("**Gambar Hasil Deteksi**")
            if record["gambar_base64"]:
                try:
                    img = base64_to_image(record["gambar_base64"])
                    st.image(img, use_container_width=True, caption="Hasil Deteksi YOLO (tersimpan)")
                except Exception:
                    st.warning("Gagal menampilkan gambar hasil deteksi.")
            else:
                st.info("Tidak ada gambar tersimpan untuk rekam medis ini.")
        st.markdown("</div>", unsafe_allow_html=True)


# ==================================================================================
# BAGIAN 6: MAIN APP & NAVIGASI SIDEBAR
# ==================================================================================
def main():
    init_db()
    load_css()

    with st.sidebar:
        st.markdown("## 🦷 CDSS Dental")
        st.caption("Clinical Decision Support System")
        st.divider()

        page = st.radio(
            "Navigasi",
            ["🏠 Beranda", "🔬 Deteksi & Anamnesis", "📋 Riwayat Pasien"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"Model YOLO: `{MODEL_PATH}`")
        if os.path.exists(MODEL_PATH):
            st.success("Model ditemukan ✅")
        else:
            st.warning("Model belum ditemukan ⚠️")
        st.caption("v1.0 · Dibuat dengan Streamlit")

    if page == "🏠 Beranda":
        page_beranda()
    elif page == "🔬 Deteksi & Anamnesis":
        page_deteksi()
    else:
        page_riwayat()


if __name__ == "__main__":
    main()
