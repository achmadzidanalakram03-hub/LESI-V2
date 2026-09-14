"""
Klinik AI RSGM — Sistem Skrining Lesi Oral berbasis YOLO
=========================================================
Versi upgrade dari prototipe awal. Perubahan utama:
  - Loading model & CSV yang tahan-error (tidak crash bila file hilang/rusak)
  - Ambang confidence & IoU yang bisa diatur dari sidebar
  - Unggah & analisis gambar secara batch (multi-file) dengan progress bar
  - Skema log lebih lengkap (ID unik, versi model, nama file) + migrasi otomatis
    dari skema lama agar CSV lama tetap kompatibel
  - Halaman "Riwayat Deteksi" dengan filter (jenis lesi, rentang tanggal,
    confidence minimum) serta ekspor CSV & Excel
  - Halaman "Analitik" baru: distribusi lesi, tren harian, distribusi confidence
  - Halaman "Referensi Lesi": ringkasan edukasi singkat per kelas lesi, lengkap
    dengan kartu hasil deteksi yang menampilkan info klinis + rekomendasi
  - Disclaimer yang jelas bahwa ini adalah alat bantu skrining, bukan diagnosis
    definitif — penting untuk aplikasi AI di ranah kesehatan
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
except ImportError:  # pustaka belum terinstal — aplikasi tetap bisa dibuka
    YOLO = None


# ============================================================
# KONFIGURASI GLOBAL
# ============================================================
APP_VERSION = "2.0"
CLINIC_NAME = "RSGM Unjani"
USER_NAME = "drg. Adinara Savero, S.KG"
USER_ROLE = "Clinical Clerkship (Koas Aktif)"

MODEL_PATH = Path("best.pt")
DB_FILE = Path("log_deteksi.csv")
DB_COLUMNS = [
    "ID", "Waktu", "Tanggal", "Lesi_Terdeteksi",
    "Confidence", "Model_Version", "Nama_File",
]

PRIMARY = "#2A9D8F"
PRIMARY_DARK = "#21867a"
ACCENT = "#E9C46A"
DANGER = "#E76F51"
DARK_TEXT = "#264653"

CONF_HIGH = 0.75
CONF_MED = 0.50

LESION_INFO = {
    "cheek biting": {
        "nama_klinis": "Morsicatio Buccarum (Cheek Biting)",
        "deskripsi": "Lesi traumatik pada mukosa bukal akibat kebiasaan menggigit "
                      "pipi berulang, umumnya tampak sebagai area putih ireguler.",
        "rekomendasi": "Edukasi pasien untuk menghentikan kebiasaan menggigit pipi; "
                        "evaluasi ulang bila lesi menetap lebih dari 2 minggu.",
        "urgensi": "Rendah",
    },
    "coated tongue": {
        "nama_klinis": "Coated Tongue (Lidah Berlapis)",
        "deskripsi": "Lapisan putih hingga kekuningan pada dorsum lidah akibat "
                      "penumpukan debris, bakteri, dan sel epitel deskuamasi.",
        "rekomendasi": "Instruksikan pembersihan lidah rutin (tongue scraper) dan "
                        "evaluasi kebersihan mulut secara umum.",
        "urgensi": "Rendah",
    },
    "karies": {
        "nama_klinis": "Karies Gigi",
        "deskripsi": "Kerusakan jaringan keras gigi akibat proses demineralisasi "
                      "oleh asam hasil metabolisme bakteri plak.",
        "rekomendasi": "Rujuk untuk pemeriksaan klinis dan radiografis lanjutan "
                        "guna menentukan rencana restorasi.",
        "urgensi": "Sedang-Tinggi",
    },
    "linea alba": {
        "nama_klinis": "Linea Alba",
        "deskripsi": "Garis putih horizontal pada mukosa bukal sepanjang bidang "
                      "oklusal, umumnya akibat tekanan/gesekan kronis dan bersifat jinak.",
        "rekomendasi": "Umumnya tidak memerlukan tatalaksana khusus; monitor bila "
                        "terjadi perubahan ukuran atau warna.",
        "urgensi": "Rendah",
    },
    "lingual varicosites": {
        "nama_klinis": "Lingual Varicosities",
        "deskripsi": "Pelebaran vena pada permukaan ventral lidah, umum ditemukan "
                      "pada individu usia lanjut, bersifat jinak.",
        "rekomendasi": "Tidak memerlukan tatalaksana khusus kecuali disertai gejala "
                        "lain; edukasi pasien mengenai sifat jinak lesi.",
        "urgensi": "Rendah",
    },
    "stain calculus": {
        "nama_klinis": "Stain & Kalkulus",
        "deskripsi": "Deposit mineral (kalkulus) dan/atau pewarnaan ekstrinsik pada "
                      "permukaan gigi akibat akumulasi plak dan faktor eksternal.",
        "rekomendasi": "Rekomendasikan scaling profesional dan evaluasi kebiasaan "
                        "oral hygiene pasien.",
        "urgensi": "Sedang",
    },
    "torus": {
        "nama_klinis": "Torus (Mandibularis/Palatinus)",
        "deskripsi": "Eksostosis tulang jinak pada mandibula atau palatum, "
                      "umumnya asimtomatik.",
        "rekomendasi": "Tidak memerlukan tindakan kecuali mengganggu fungsi "
                        "(bicara, protesa) atau membesar signifikan.",
        "urgensi": "Rendah",
    },
    "ulkus traumatikus": {
        "nama_klinis": "Ulkus Traumatikus",
        "deskripsi": "Lesi ulseratif pada mukosa oral akibat trauma mekanis, "
                      "termal, atau kimiawi, biasanya sembuh spontan.",
        "rekomendasi": "Evaluasi ulang bila tidak sembuh dalam 10-14 hari untuk "
                        "menyingkirkan diagnosis banding lain.",
        "urgensi": "Sedang",
    },
}


# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# --- KUSTOMISASI CSS (UI/UX MODERN) ---
st.markdown("""
    <style>
    .stApp { background-color: #F4F7F6; }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2A9D8F;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #264653; margin: 0; }
    .metric-label {
        font-size: 0.9rem; color: #6c757d;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .image-container {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E9ECEF;
    }
    .lesion-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #2A9D8F;
    }
    .stButton>button {
        background-color: #2A9D8F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21867a;
        box-shadow: 0 4px 8px rgba(42, 157, 143, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI BANTU (DATABASE, MODEL, TAMPILAN)
# ============================================================
def init_db() -> None:
    """Pastikan file CSV log ada dan skemanya sudah sesuai DB_COLUMNS terkini.

    Jika file lama memakai skema berbeda (mis. dari versi aplikasi sebelumnya),
    file akan ditulis ulang dengan kolom yang selaras. Ini penting agar proses
    `append_log` (append tanpa header) tidak pernah menambahkan baris dengan
    jumlah kolom yang berbeda dari file yang sudah ada — kondisi yang akan
    merusak CSV dan membuat `pd.read_csv` gagal total pada rerun berikutnya.
    Jika file ternyata korup/tidak bisa diparse, file lama dicadangkan
    (`*_backup.csv`) alih-alih menimpa/menghapus data begitu saja.
    """
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
    """Muat log deteksi (memicu migrasi skema otomatis via init_db)."""
    init_db()
    try:
        df = pd.read_csv(DB_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=DB_COLUMNS)
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    return df[DB_COLUMNS]


def append_log(records: list) -> None:
    """Tambahkan baris baru ke CSV log (skema selalu diselaraskan lebih dulu)."""
    if not records:
        return
    init_db()
    df_new = pd.DataFrame(records)[DB_COLUMNS]
    header = (not DB_FILE.exists()) or DB_FILE.stat().st_size == 0
    df_new.to_csv(DB_FILE, mode="a", header=header, index=False)


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Muat model YOLO. Mengembalikan None (bukan melempar exception) bila gagal,
    supaya sisa aplikasi tetap bisa dibuka dan menampilkan pesan yang jelas."""
    if YOLO is None or not path.exists():
        return None
    try:
        return YOLO(str(path))
    except Exception:
        return None


def confidence_badge(conf: float) -> str:
    if conf >= CONF_HIGH:
        color = PRIMARY
    elif conf >= CONF_MED:
        color = ACCENT
    else:
        color = DANGER
    return (
        f"<span style='background:{color}; color:white; padding:2px 10px; "
        f"border-radius:12px; font-size:0.8rem; font-weight:bold;'>{conf*100:.1f}%</span>"
    )


def get_lesion_info(nama: str) -> dict:
    info = LESION_INFO.get(str(nama).lower().strip())
    if info:
        return info
    return {
        "nama_klinis": str(nama).title(),
        "deskripsi": "Informasi klinis belum tersedia untuk kelas ini.",
        "rekomendasi": "Konsultasikan dengan dokter gigi penanggung jawab.",
        "urgensi": "Tidak diketahui",
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.markdown(f"### {CLINIC_NAME}")
    st.caption("AI Dental Vision System")
    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        ["Dashboard", "Riwayat Deteksi", "Analitik", "Referensi Lesi", "Tentang Aplikasi", "Kontak"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan Deteksi")
    conf_threshold = st.slider("Ambang Confidence", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("Ambang IoU (NMS)", 0.05, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**Profil Pengguna**")
    st.markdown(f"👨‍⚕️ {USER_NAME}")
    st.caption(f"Status: {USER_ROLE}")
    st.markdown("---")
    st.caption(f"Versi Aplikasi: {APP_VERSION}")

model = load_model(MODEL_PATH)


# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard":
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.title("Sistem Skrining Lesi Oral")
        st.markdown(
            f"<p style='color:#6c757d;'>Tanggal Hari Ini: {datetime.now().strftime('%d %B %Y')}</p>",
            unsafe_allow_html=True,
        )
    with col_header2:
        badge_color = PRIMARY if model else DANGER
        badge_text = "Model Aktif" if model else "Model Tidak Aktif"
        st.markdown(
            f"<div style='text-align:right; padding-top:20px;'>"
            f"<span style='background:{badge_color}; color:white; padding:4px 12px; "
            f"border-radius:20px; font-size:0.8rem;'>{badge_text}</span></div>",
            unsafe_allow_html=True,
        )

    if model is None:
        st.error(
            "⚠️ Model YOLO ('best.pt') tidak ditemukan atau gagal dimuat. Pastikan "
            "file `best.pt` berada di direktori yang sama dengan aplikasi ini dan "
            "pustaka `ultralytics` sudah terinstal (`pip install ultralytics`)."
        )

    df_log = load_log()
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0%"

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Pemeriksaan AI</p>
            <p class="metric-value">{total_deteksi}</p>
            <p style="color:{PRIMARY}; font-size:0.8rem; margin:0;">▲ {deteksi_hari_ini} deteksi hari ini</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Rata-rata Confidence</p>
            <p class="metric-value">{avg_conf}</p>
            <p style="color:{ACCENT}; font-size:0.8rem; margin:0;">Berdasarkan YOLO</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        status_text = "Sinkron" if model else "Model Error"
        status_color = PRIMARY if model else DANGER
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Status Integrasi</p>
            <p class="metric-value">{status_text}</p>
            <p style="color:{status_color}; font-size:0.8rem; margin:0;">Database Real-time</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<h4 style='color:#264653; margin-top:20px;'>Grafik Distribusi Lesi</h4>", unsafe_allow_html=True)
    if total_deteksi > 0:
        chart_data = df_log.groupby(["Tanggal", "Lesi_Terdeteksi"]).size().unstack(fill_value=0)
        st.area_chart(chart_data, use_container_width=True)
    else:
        st.info("Menunggu data deteksi pertama masuk ke dalam sistem.")

    st.markdown("---")
    st.markdown("<h4 style='color:#264653;'>Modul Analisis Citra Klinis</h4>", unsafe_allow_html=True)
    st.caption(
        "⚠️ Hasil deteksi AI merupakan alat bantu skrining awal dan **bukan diagnosis "
        "definitif**. Konfirmasi klinis oleh dokter gigi tetap diperlukan sebelum "
        "pengambilan keputusan perawatan."
    )

    uploaded_files = st.file_uploader(
        "Seret dan lepas (drag & drop) satu atau beberapa foto intraoral pasien di sini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} gambar siap dianalisis**")
        preview_cols = st.columns(min(len(uploaded_files), 5))
        for i, f in enumerate(uploaded_files[:5]):
            with preview_cols[i]:
                st.image(f, use_container_width=True, caption=f.name)
        if len(uploaded_files) > 5:
            st.caption(f"+{len(uploaded_files) - 5} gambar lainnya tidak ditampilkan pada pratinjau")

        analyze_btn = st.button(
            "🔬 Mulai Analisis YOLO", use_container_width=True, disabled=(model is None)
        )

        if analyze_btn and model is not None:
            progress = st.progress(0, text="Memulai analisis...")
            all_new_records = []
            total_lesi = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                progress.progress(idx / len(uploaded_files), text=f"Memindai {uploaded_file.name}...")

                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception:
                    st.warning(f"⚠️ Gagal membuka gambar: {uploaded_file.name} (format tidak didukung/rusak).")
                    continue

                try:
                    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
                except Exception as e:
                    st.error(f"⚠️ Inferensi gagal untuk {uploaded_file.name}: {e}")
                    continue

                res_plotted = results[0].plot()
                boxes = results[0].boxes
                waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tanggal_sekarang = datetime.now().strftime("%Y-%m-%d")

                detections = []
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    nama_lesi = model.names[class_id]
                    detections.append((nama_lesi, conf_score))
                    all_new_records.append({
                        "ID": str(uuid.uuid4())[:8],
                        "Waktu": waktu_sekarang,
                        "Tanggal": tanggal_sekarang,
                        "Lesi_Terdeteksi": nama_lesi,
                        "Confidence": round(conf_score, 4),
                        "Model_Version": MODEL_PATH.name,
                        "Nama_File": uploaded_file.name,
                    })
                total_lesi += len(detections)

                with st.expander(
                    f"📸 {uploaded_file.name} — {len(detections)} lesi terdeteksi",
                    expanded=(len(uploaded_files) == 1),
                ):
                    col_img1, col_img2 = st.columns(2, gap="large")
                    with col_img1:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.markdown("**📸 Citra Klinis Masukan**")
                        st.image(image, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col_img2:
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.markdown("**🧬 Hasil Pemetaan Bounding Box**")
                        st.image(res_plotted, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if not detections:
                        st.info(
                            "Jaringan tampak sehat / tidak ada lesi spesifik yang "
                            "terdeteksi pada ambang confidence saat ini."
                        )
                    else:
                        st.markdown("##### Rincian Temuan")
                        for nama_lesi, conf_score in sorted(detections, key=lambda x: -x[1]):
                            info = get_lesion_info(nama_lesi)
                            st.markdown(
                                f"""<div class='lesion-card'>
                                <b>{info['nama_klinis']}</b> {confidence_badge(conf_score)}<br>
                                <span style='font-size:0.85rem; color:#495057;'>{info['deskripsi']}</span><br>
                                <span style='font-size:0.85rem; color:{DARK_TEXT};'>
                                <b>Rekomendasi:</b> {info['rekomendasi']} &nbsp;|&nbsp; <b>Urgensi:</b> {info['urgensi']}
                                </span></div>""",
                                unsafe_allow_html=True,
                            )

            progress.progress(1.0, text="Selesai.")
            append_log(all_new_records)

            if total_lesi > 0:
                st.success(f"✅ Analisis selesai. {total_lesi} lesi terdeteksi dan log berhasil diperbarui ke database.")
            else:
                st.info("Analisis selesai. Tidak ada lesi yang tercatat pada ambang confidence saat ini.")


# ============================================================
# HALAMAN: RIWAYAT DETEKSI
# ============================================================
elif menu == "Riwayat Deteksi":
    st.title("Riwayat Data Pasien")
    df_log = load_log()

    if df_log.empty:
        st.info("Belum ada data deteksi yang tercatat.")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            lesi_options = sorted(df_log["Lesi_Terdeteksi"].dropna().unique().tolist())
            selected_lesi = st.multiselect("Filter Jenis Lesi", lesi_options, default=lesi_options)

        with col_f2:
            tanggal_series = pd.to_datetime(df_log["Tanggal"], errors="coerce").dropna()
            if not tanggal_series.empty:
                min_date, max_date = tanggal_series.min(), tanggal_series.max()
                date_range = st.date_input(
                    "Rentang Tanggal", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date,
                )
            else:
                date_range = None

        with col_f3:
            min_conf = st.slider("Confidence Minimum", 0.0, 1.0, 0.0, 0.05)

        mask = df_log["Lesi_Terdeteksi"].isin(selected_lesi) & (df_log["Confidence"].fillna(0) >= min_conf)

        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            tgl_dt = pd.to_datetime(df_log["Tanggal"], errors="coerce")
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            mask &= (tgl_dt >= start) & (tgl_dt <= end)

        df_filtered = df_log[mask]

        st.dataframe(df_filtered, use_container_width=True)
        st.caption(f"Menampilkan {len(df_filtered)} dari {len(df_log)} total catatan.")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Unduh CSV",
                data=df_filtered.to_csv(index=False).encode("utf-8"),
                file_name="Laporan_Deteksi_Lesi.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_filtered.to_excel(writer, index=False, sheet_name="Riwayat Deteksi")
                st.download_button(
                    "⬇️ Unduh Excel",
                    data=buffer.getvalue(),
                    file_name="Laporan_Deteksi_Lesi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("Instal pustaka `openpyxl` (`pip install openpyxl`) untuk mengaktifkan ekspor Excel.")


# ============================================================
# HALAMAN: ANALITIK
# ============================================================
elif menu == "Analitik":
    st.title("Analitik & Performa")
    df_log = load_log()

    if df_log.empty:
        st.info("Belum ada data untuk dianalisis.")
    else:
        st.markdown("#### Distribusi Jenis Lesi")
        st.bar_chart(df_log["Lesi_Terdeteksi"].value_counts())

        st.markdown("#### Tren Deteksi Harian")
        st.line_chart(df_log.groupby("Tanggal").size())

        st.markdown("#### Distribusi Confidence Score")
        conf_bins = pd.cut(df_log["Confidence"].dropna(), bins=10)
        st.bar_chart(conf_bins.value_counts().sort_index().rename(lambda x: str(x)))

        st.markdown("#### Ringkasan Statistik")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Deteksi", len(df_log))
        col_s2.metric("Jenis Lesi Unik", df_log["Lesi_Terdeteksi"].nunique())
        col_s3.metric("Confidence Tertinggi", f"{df_log['Confidence'].max()*100:.1f}%")
        col_s4.metric("Confidence Terendah", f"{df_log['Confidence'].min()*100:.1f}%")

        st.caption(
            "Catatan: visualisasi *confusion matrix* dan kurva presisi-recall "
            "memerlukan data ground-truth dari proses pelatihan/validasi model, "
            "dan dapat ditambahkan sebagai iterasi berikutnya menggunakan hasil "
            "`model.val()` dari Ultralytics."
        )


# ============================================================
# HALAMAN: REFERENSI LESI
# ============================================================
elif menu == "Referensi Lesi":
    st.title("Referensi Klinis Lesi Oral")
    st.caption(
        "Rangkuman edukasi singkat untuk kelas lesi yang dikenali oleh model. "
        "Bukan pengganti literatur akademik atau penilaian klinis dokter gigi."
    )
    for info in LESION_INFO.values():
        with st.expander(info["nama_klinis"]):
            st.write(info["deskripsi"])
            st.markdown(f"**Rekomendasi:** {info['rekomendasi']}")
            st.markdown(f"**Tingkat Urgensi:** {info['urgensi']}")


# ============================================================
# HALAMAN: TENTANG APLIKASI
# ============================================================
elif menu == "Tentang Aplikasi":
    st.title("Tentang Aplikasi")
    tab_fitur, tab_tentang, tab_project = st.tabs(["Fitur", "Tentang", "Project"])

    with tab_fitur:
        st.write(
            "Sistem inferensi ini didukung oleh arsitektur YOLO yang dioptimalkan "
            "untuk mendeteksi *bounding box* anomali oral. Pemantauan metrik dan "
            "distribusi gambar langsung disinkronkan ke dalam *dashboard*, dengan "
            "ambang confidence dan IoU yang dapat disesuaikan pengguna."
        )
    with tab_tentang:
        st.write(
            "Aplikasi skrining ini dirancang untuk memfasilitasi pengambilan "
            "keputusan klinis dan mendukung kolaborasi interprofesional di "
            "lingkungan layanan kesehatan tingkat pertama maupun rumah sakit "
            "pendidikan."
        )
        st.caption(
            "⚠️ Aplikasi ini adalah alat bantu skrining berbasis AI dan tidak "
            "dimaksudkan untuk menggantikan pemeriksaan klinis oleh dokter gigi."
        )
    with tab_project:
        st.write(
            "Area ini didedikasikan untuk menampilkan visualisasi *confusion "
            "matrix*, kurva presisi-recall, dan performa uji hipotesis dari "
            "iterasi model pelatihan. Lihat halaman **Analitik** untuk statistik "
            "berbasis data yang sudah terkumpul."
        )


# ============================================================
# HALAMAN: KONTAK
# ============================================================
elif menu == "Kontak":
    st.title("Hubungi Pengembang")
    st.write("Untuk kebutuhan teknis dan kalibrasi model, silakan hubungi tim administrator klinis.")
