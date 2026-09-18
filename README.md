# Mammouth — My Assistant in Mouth Health

Platform skrining lesi rongga mulut berbasis computer vision. Proyek independen, v6.0.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Letakkan berkas bobot model di folder yang sama dengan `app.py`:

| Arsitektur | Nama berkas yang dicari |
|---|---|
| YOLOv8 | `best.pt`, `yolov8_best.pt` |
| YOLOv11 | `yolov11_best.pt`, `yolo11_best.pt`, `best.pt` |
| YOLOv12 | `yolov12_best.pt`, `yolo12_best.pt`, `best.pt` |

Tanpa bobot, aplikasi tetap bisa dijelajahi lewat **mode demo** (Pengaturan → Model). Pemeriksaan
mode demo ditandai dan bisa disembunyikan dari analitik.

## Akun dan pemisahan data

- Registrasi terbuka lewat tab **Buat akun** di halaman masuk. Akun pertama di server otomatis menjadi administrator.
- Kata sandi disimpan sebagai PBKDF2-HMAC-SHA256, 200.000 iterasi, dengan salt acak per akun.
- Setiap baris pasien, pemeriksaan, dan deteksi membawa `user_id`. Semua kueri menyaring berdasarkan
  kolom itu, jadi satu akun tidak dapat membuka data akun lain dari dalam aplikasi.
- Citra disimpan terpisah per akun di `mammouth_data/images/<user_id>/`.
- Administrator hanya melihat daftar akun dan jumlah datanya, bukan isi rekam medisnya.

## Penyimpanan

Semua data berada di folder `mammouth_data/` (ubah lewat variabel lingkungan `MAMMOUTH_DATA_DIR`):

```
mammouth_data/
├── mammouth.db          # users, patients, exams, detections, activity
└── images/<user_id>/    # citra asli dan citra beranotasi
```

Basis data versi 5 (`mammouth.db` di folder aplikasi, tabel `users` + `emr_logs`) dipindahkan
otomatis sekali jalan saat aplikasi pertama kali dibuka. Akun lama tetap bisa masuk dengan kata
sandi lamanya, dan hash-nya dinaikkan ke PBKDF2 saat login berhasil.

## Fitur

- Pasien, pemeriksaan, dan deteksi sebagai tabel terpisah dengan riwayat per pasien
- Anamnesis OLD CARTS yang memengaruhi sintesis temuan dan tingkat prioritas
- Unggah banyak citra sekaligus atau ambil langsung dari kamera intraoral
- Penyesuaian kecerahan, kontras, ketajaman, dan auto-kontras dengan pratinjau langsung
- Laporan HTML siap cetak per pemeriksaan
- Ekspor CSV, Excel, dan arsip ZIP berisi seluruh data dan citra akun
- Analitik: frekuensi kelas, sebaran keyakinan, aktivitas harian, skala nyeri
- Ensiklopedia delapan kelas lesi dengan etiologi, tatalaksana, diagnosis banding, dan tanda bahaya
- Tema terang dan gelap, tersimpan per akun

## Untuk pemakaian klinis nyata

Aplikasi ini alat bantu penapisan, bukan alat diagnosis. Selain itu, sebelum dipakai dengan data
pasien sungguhan: jalankan di balik HTTPS, aktifkan enkripsi disk pada server, dan siapkan
pencadangan berkala untuk folder `mammouth_data/`.
