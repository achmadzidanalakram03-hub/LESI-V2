import base64
from datetime import datetime
from io import BytesIO

import numpy as np
import os
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klinik AI RSGM", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# DESAIN SYSTEM: FONT, WARNA, ANIMASI
# ============================================================
# Palet dikembangkan dari indigo yang sudah kamu pakai (#5B65DC), dibuat lebih
# tegas untuk kontras & hierarki, ditambah teal untuk status "aman/sukses" agar
# tidak bentrok dengan warna brand utama.
# Font: Plus Jakarta Sans (judul & body — hangat tapi tetap presisi untuk
# konteks klinis) + JetBrains Mono khusus untuk ANGKA (confidence, jam, ID)
# supaya data yang dibaca mesin terasa berbeda dari teks yang ditulis manusia —
# konvensi umum di dashboard instrumen/analitik.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --ink: #10142C;
        --muted: #676C93;
        --primary: #4B4FE0;
        --primary-dark: #2F32A8;
        --primary-tint: #EEF0FD;
        --teal: #0FA88A;
        --teal-tint: #E7F8F4;
        --amber: #C8860A;
        --amber-tint: #FBF1DD;
        --coral: #D8453A;
        --coral-tint: #FBEAE8;
        --bg: #F5F6FB;
        --surface: #FFFFFF;
        --border: #E5E7F5;
    }

    html, body, [class*="st-"], [data-baseweb] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }

    .stApp { background-color: var(--bg); }
    .stMarkdown, .stText, h1, h2, h3, h4, h5, p, label { color: var(--ink); }

    /* Fokus keyboard tetap terlihat jelas (aksesibilitas) */
    a:focus-visible, button:focus-visible, [role="radio"]:focus-visible,
    [tabindex]:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }

    /* Scrollbar halus */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: #C7CBF2; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    /* ============ SIDEBAR ============ */
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 9px 14px;
        border-radius: 10px;
        transition: background-color 0.15s ease, transform 0.1s ease;
        width: 100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: var(--primary-tint);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background-color: var(--primary);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p {
        color: white !important;
        font-weight: 600;
    }

    .brand-row { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
    .brand-title { font-weight: 800; font-size: 1.15rem; color: var(--ink); margin: 0; line-height: 1.1; }
    .brand-sub { color: var(--primary); font-size: 0.78rem; font-weight: 600; margin: 2px 0 0 0; }

    .profile-chip {
        display: flex; align-items: center; gap: 10px;
        background: var(--primary-tint);
        border-radius: 12px;
        padding: 10px 12px;
    }
    .profile-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: var(--primary); color: white;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
    }
    .profile-name { font-weight: 700; font-size: 0.88rem; margin: 0; color: var(--ink); }
    .profile-role { font-size: 0.74rem; color: var(--muted); margin: 0; }

    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.76rem; font-weight: 600;
        padding: 4px 10px; border-radius: 20px;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; }
    .status-on { background: var(--teal-tint); color: var(--teal); }
    .status-on .status-dot { background: var(--teal); }
    .status-off { background: var(--coral-tint); color: var(--coral); }
    .status-off .status-dot { background: var(--coral); }

    @media (prefers-reduced-motion: no-preference) {
        .status-on .status-dot { animation: pulse-dot 1.8s ease-in-out infinite; }
    }
    @keyframes pulse-dot {
        0%, 100% { box-shadow: 0 0 0 0 rgba(15, 168, 138, 0.45); }
        50% { box-shadow: 0 0 0 5px rgba(15, 168, 138, 0); }
    }

    /* ============ HERO / HEADER ============ */
    .hero-enter { animation: none; }
    @media (prefers-reduced-motion: no-preference) {
        .hero-enter { animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .page-title { font-weight: 800; font-size: 2rem; margin: 0 0 2px 0; letter-spacing: -0.02em; }
    .page-date { color: var(--muted); font-size: 0.92rem; margin: 0 0 18px 0; }

    /* ============ KPI CARDS (hierarki, bukan 3 kembar) ============ */
    .kpi-hero {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 16px;
        padding: 22px 22px;
        color: white;
        box-shadow: 0 10px 24px -12px rgba(75, 79, 224, 0.55);
    }
    .kpi-hero .metric-label { color: rgba(255,255,255,0.75); }
    .kpi-hero .metric-value { color: white; font-family: 'JetBrains Mono', monospace; }
    .kpi-hero .metric-sub { color: rgba(255,255,255,0.85); }

    .metric-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px;
        height: 100%;
    }
    .metric-label {
        font-size: 0.8rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.06em;
        margin: 0 0 8px 0; font-weight: 600;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: var(--ink);
        margin: 0; line-height: 1.15; font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub { font-size: 0.82rem; font-weight: 600; margin: 8px 0 0 0; color: var(--primary); }

    /* ============ TABS (Upload / Kamera) ============ */
    [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    [data-baseweb="tab"] {
        font-weight: 600; color: var(--muted);
        padding: 10px 4px; margin-right: 20px;
    }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--primary); }
    [data-baseweb="tab-highlight"] { background-color: var(--primary) !important; height: 2.5px; }

    /* ============ UPLOADER & KAMERA ============ */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--primary-tint);
        border: 1.5px dashed #B9BEF0;
        border-radius: 14px;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary);
        background-color: #E4E7FC;
    }
    [data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img {
        border-radius: 14px;
    }

    /* ============ TOMBOL ============ */
    .stButton>button {
        background-color: var(--primary);
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 0.98rem;
        width: 100%;
        transition: transform 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease;
    }
    .stButton>button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 8px 18px -8px rgba(75, 79, 224, 0.55);
        transform: translateY(-1px);
    }
    .stButton>button:active { transform: translateY(0); }
    .stButton>button:disabled { background-color: #C7CBF2; color: #8489C9 !important; }

    /* ============ PANEL GAMBAR + BINGKAI "VIEWFINDER" ============ */
    /* Sudut siku-siku ini sengaja meniru bounding box yang digambar YOLO —
       satu momen visual yang benar-benar terkait dengan fungsi aplikasi,
       bukan sekadar dekorasi. */
    .image-panel { margin-bottom: 14px; }
    .image-panel-label { font-weight: 700; font-size: 0.92rem; margin: 0 0 10px 0; color: var(--ink); }
    .viewfinder {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 10px;
    }
    .vf-img { width: 100%; display: block; border-radius: 8px; }
    .vf-corner {
        position: absolute; width: 22px; height: 22px;
        border-color: var(--primary); z-index: 2; opacity: 0.85;
    }
    .vf-tl { top: 6px; left: 6px; border-top: 3px solid; border-left: 3px solid; border-top-left-radius: 6px; }
    .vf-tr { top: 6px; right: 6px; border-top: 3px solid; border-right: 3px solid; border-top-right-radius: 6px; }
    .vf-bl { bottom: 6px; left: 6px; border-bottom: 3px solid; border-left: 3px solid; border-bottom-left-radius: 6px; }
    .vf-br { bottom: 6px; right: 6px; border-bottom: 3px solid; border-right: 3px solid; border-bottom-right-radius: 6px; }
    @media (prefers-reduced-motion: no-preference) {
        .vf-active .vf-corner { animation: vf-pulse 1.4s ease-in-out infinite; }
        .vf-result .viewfinder { animation: fadeSlideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) both; }
    }
    @keyframes vf-pulse {
        0%, 100% { opacity: 0.55; }
        50% { opacity: 1; }
    }

    /* ============ KARTU HASIL & ALERT KUSTOM ============ */
    .lesion-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .conf-badge {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; font-size: 0.78rem;
        padding: 2px 9px; border-radius: 20px; color: white;
    }
    .app-alert {
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 0.92rem;
        display: flex; gap: 10px; align-items: flex-start;
        margin: 10px 0;
    }
    .app-alert.info { background: var(--primary-tint); color: var(--primary-dark); }
    .app-alert.success { background: var(--teal-tint); color: #0B7A64; }
    .app-alert.warning { background: var(--amber-tint); color: #8A620A; }
    .app-alert.error { background: var(--coral-tint); color: #A6362D; }

    .content-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 26px;
    }

    /* ============ MOBILE ============ */
    @media (max-width: 768px) {
        .page-title { font-size: 1.55rem; }
        .kpi-hero, .metric-card { padding: 16px; border-radius: 14px; }
        .metric-value { font-size: 1.55rem; }
        .stButton>button { padding: 14px; font-size: 1rem; }
        .content-card { padding: 18px; }
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# KONFIGURASI DATA & MODEL
# ============================================================
DB_FILE = "log_deteksi.csv"
USER_NAME = "Adinara Savero, S.KG"
USER_ROLE = "Mahasiswa FKG (Aktif)"


def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["Waktu", "Tanggal", "Lesi_Terdeteksi", "Confidence"])
        df.to_csv(DB_FILE, index=False)


init_db()


@st.cache_resource(show_spinner=False)
def load_model():
    """Model dimuat dengan aman — kalau best.pt hilang/rusak, aplikasi tetap
    terbuka dan menampilkan status yang jelas, bukan layar error mentah."""
    try:
        return YOLO('best.pt')
    except Exception:
        return None


model = load_model()


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_viewfinder(label: str, img: Image.Image, icon: str = "📸", active: bool = False, result: bool = False):
    """Menampilkan gambar dalam bingkai bersudut ala viewfinder — motif yang
    sengaja meniru bounding box YOLO, dirender sebagai satu blok HTML utuh
    (base64) supaya bingkainya benar-benar membungkus gambar, bukan sekadar
    div kosong yang berdiri sendiri."""
    b64 = image_to_base64(img)
    vf_class = "vf-active" if active else ""
    wrapper_class = "vf-result" if result else ""
    st.markdown(
        f"""
        <div class="image-panel {wrapper_class}">
            <p class="image-panel-label">{icon} {label}</p>
            <div class="viewfinder {vf_class}">
                <span class="vf-corner vf-tl"></span>
                <span class="vf-corner vf-tr"></span>
                <span class="vf-corner vf-bl"></span>
                <span class="vf-corner vf-br"></span>
                <img src="data:image/png;base64,{b64}" class="vf-img" />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert(message: str, kind: str = "info", icon: str = "ℹ️"):
    """Alert bergaya sendiri (bukan kotak biru/oranye bawaan Streamlit) supaya
    warnanya konsisten dengan palet aplikasi."""
    st.markdown(
        f"""<div class="app-alert {kind}"><span>{icon}</span><span>{message}</span></div>""",
        unsafe_allow_html=True,
    )


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#0FA88A"
    if conf >= 0.5:
        return "#C8860A"
    return "#D8453A"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="34" />
            <div>
                <p class="brand-title">RSGM Unjani</p>
                <p class="brand-sub">AI Dental Vision System</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        ["Dashboard", "Riwayat Deteksi", "Feature", "About", "Project", "Contact"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    initials = "".join([w[0] for w in USER_NAME.split()[:2]]).upper()
    st.markdown(
        f"""
        <div class="profile-chip">
            <div class="profile-avatar">{initials}</div>
            <div>
                <p class="profile-name">{USER_NAME}</p>
                <p class="profile-role">{USER_ROLE}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if model is not None:
        st.markdown(
            """<span class="status-pill status-on"><span class="status-dot"></span>Model Aktif</span>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<span class="status-pill status-off"><span class="status-dot"></span>Model Tidak Aktif</span>""",
            unsafe_allow_html=True,
        )


# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if menu == "Dashboard":
    st.markdown(
        f"""
        <div class="hero-enter">
            <p class="page-title">Sistem Skrining Lesi Oral</p>
            <p class="page-date">Tanggal Hari Ini: {datetime.now().strftime('%d %B %Y')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        alert(
            "Model YOLO (<code>best.pt</code>) tidak ditemukan atau gagal dimuat. "
            "Pastikan file berada di folder yang sama dengan aplikasi ini.",
            kind="error", icon="⚠️",
        )

    df_log = pd.read_csv(DB_FILE)
    total_deteksi = len(df_log)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    deteksi_hari_ini = len(df_log[df_log["Tanggal"] == hari_ini])
    avg_conf = f"{df_log['Confidence'].mean() * 100:.1f}%" if total_deteksi > 0 else "0%"

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-hero">
            <p class="metric-label">Total Pemeriksaan AI</p>
            <p class="metric-value">{total_deteksi}</p>
            <p class="metric-sub">▲ {deteksi_hari_ini} pasien hari ini</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Rata-rata Confidence</p>
            <p class="metric-value">{avg_conf}</p>
            <p class="metric-sub">Berdasarkan YOLOv11</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status_text = "Sinkron" if model is not None else "Model Error"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Status Integrasi</p>
            <p class="metric-value" style="font-family:'Plus Jakarta Sans',sans-serif;">{status_text}</p>
            <p class="metric-sub">Database Real-time Aktif</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h4 style='margin-top: 26px;'>Grafik Distribusi Lesi</h4>", unsafe_allow_html=True)
    if total_deteksi > 0:
        chart_data = df_log.groupby(['Tanggal', 'Lesi_Terdeteksi']).size().unstack(fill_value=0)
        st.area_chart(chart_data, use_container_width=True, color=["#4B4FE0", "#0FA88A", "#C8860A", "#D8453A"])
    else:
        alert("Menunggu data deteksi pertama masuk ke dalam sistem.", kind="info", icon="🕒")

    st.markdown("---")
    st.markdown("<h4>Modul Analisis Citra Klinis</h4>", unsafe_allow_html=True)

    tab_unggah, tab_kamera = st.tabs(["Upload", "Kamera"])

    image = None

    with tab_unggah:
        uploaded_file = st.file_uploader("Pilih foto intraoral dari penyimpanan perangkat", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

    with tab_kamera:
        camera_file = st.camera_input("Ambil gambar lesi secara langsung (gunakan ikon rotate kamera bawaan HP untuk opsi depan/belakang)")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')

    if image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_img1, col_img2 = st.columns(2, gap="large")

        with col_img1:
            render_viewfinder("Citra Klinis Masukan", image, icon="📸", active=True)
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button('Mulai Analisis YOLO', use_container_width=True, disabled=(model is None))

        if analyze_btn and model is not None:
            with st.spinner('Memindai anomali dental...'):
                results = model(image)
                res_plotted_bgr = results[0].plot()
                res_plotted = Image.fromarray(res_plotted_bgr[:, :, ::-1])  # BGR -> RGB

                boxes = results[0].boxes
                if len(boxes) > 0:
                    new_records = []
                    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tanggal_sekarang = datetime.now().strftime("%Y-%m-%d")

                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        conf_score = float(box.conf[0].item())
                        nama_lesi = model.names[class_id]

                        new_records.append({
                            "Waktu": waktu_sekarang,
                            "Tanggal": tanggal_sekarang,
                            "Lesi_Terdeteksi": nama_lesi,
                            "Confidence": round(conf_score, 2)
                        })

                    df_new = pd.DataFrame(new_records)
                    df_new.to_csv(DB_FILE, mode='a', header=False, index=False)

                with col_img2:
                    render_viewfinder("Hasil Pemetaan Bounding Box", res_plotted, icon="🧬", result=True)

                    if len(boxes) == 0:
                        alert("Jaringan sehat / tidak ada lesi yang terdeteksi secara spesifik.", kind="warning", icon="🦷")
                    else:
                        alert("Log pasien berhasil diperbarui ke database.", kind="success", icon="✅")
                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            conf_score = float(box.conf[0].item())
                            nama_lesi = model.names[class_id]
                            color = confidence_color(conf_score)
                            st.markdown(
                                f"""<div class="lesion-card">
                                <b>{nama_lesi.title()}</b>
                                <span class="conf-badge" style="background:{color};">{conf_score*100:.1f}%</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )


elif menu == "Riwayat Deteksi":
    st.markdown("<p class='page-title'>Riwayat Data Pasien</p>", unsafe_allow_html=True)
    df_log = pd.read_csv(DB_FILE)
    if df_log.empty:
        alert("Belum ada data deteksi yang tercatat.", kind="info", icon="🗂️")
    else:
        st.dataframe(df_log, use_container_width=True)
        with open(DB_FILE, "rb") as file:
            st.download_button("⬇️ Unduh Laporan CSV", data=file, file_name="Laporan_Deteksi_Lesi.csv", mime="text/csv")

elif menu == "Feature":
    st.markdown("<p class='page-title'>Fitur Sistem</p>", unsafe_allow_html=True)
    st.markdown(
        """<div class="content-card">Sistem inferensi didukung arsitektur YOLOv11. Pemantauan metrik dan
        distribusi disinkronkan ke dalam dashboard secara real-time.</div>""",
        unsafe_allow_html=True,
    )

elif menu == "About":
    st.markdown("<p class='page-title'>Tentang Aplikasi</p>", unsafe_allow_html=True)
    st.markdown(
        """<div class="content-card">Aplikasi skrining ini dirancang untuk memfasilitasi pengambilan keputusan
        klinis dan mendukung kolaborasi interprofesional.</div>""",
        unsafe_allow_html=True,
    )

elif menu == "Project":
    st.markdown("<p class='page-title'>Project Overview</p>", unsafe_allow_html=True)
    st.markdown(
        """<div class="content-card">Dokumentasi <i>confusion matrix</i> dan performa uji hipotesis dari
        iterasi model pelatihan akan ditampilkan di sini.</div>""",
        unsafe_allow_html=True,
    )

elif menu == "Contact":
    st.markdown("<p class='page-title'>Hubungi Pengembang</p>", unsafe_allow_html=True)
    st.markdown(
        """<div class="content-card">Untuk kebutuhan kalibrasi model, silakan hubungi tim administrator klinis.</div>""",
        unsafe_allow_html=True,
    )
