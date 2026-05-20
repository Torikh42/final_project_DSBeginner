# Fragrance Matchmaker — Implementation Plan

> **Informasi Umum & Deadline:**
> *   **Deadline Tugas:** Sabtu, 23 Mei 2026 pukul 23:59 WIB
> *   **Anggota Tim:**
>     1. Torikh Abdullah Naser
>     2. Alif Ilham Rhamadan
>     3. Hasan Shofiyyurrahman
> *   **Topik:** Sistem Rekomendasi Parfum (Content-Based Filtering)
> *   **Repositori Git:** `https://github.com/Torikh42/final_project_DSBeginner.git`

---

## 👥 Pembagian Tugas (Task Division)

Untuk memaksimalkan efisiensi dan keahlian masing-masing anggota tim, berikut adalah pembagian tugas yang terstruktur:

1. **Torikh Abdullah Naser (Project Coordinator & Integration)**
   - **Tugas:** Project setup, EDA & visualisasi awal, koordinasi integrasi sistem, peninjauan unit test, deployment ke Streamlit Cloud, dan penyusunan berkas akhir (`README.md` & `link_aplikasi.txt`).
   - **Kontribusi:** Phase 1, Phase 2, Phase 3 (selesai), Phase 6 (bersama tim).

2. **Hasan Shofiyyurrahman (ML Backend & Recommender Engine)**
   - **Tugas:** Menulis dan menguji engine sistem rekomendasi. Hasan akan mengimplementasikan model Content-Based Filtering menggunakan CountVectorizer & Cosine Similarity serta menulis unit test (`pytest`) untuk mengevaluasi akurasi dan robust-ness model.
   - **Kontribusi:** Phase 4 (Recommender Engine & TDD Evaluation).

3. **Alif Ilham Rhamadan (Frontend UI & Streamlit Design)**
   - **Tugas:** Merancang antarmuka aplikasi Streamlit yang premium, mengimplementasikan Custom CSS (glassmorphism/dark mode), memetakan input sidebar (kategori, target audience, longevity) secara dinamis agar tidak terjadi *dead-ends*, dan menyambungkan input user dengan engine rekomendasi buatan Hasan.
   - **Kontribusi:** Phase 5 (Premium Streamlit Web UI).

---

## 📁 Struktur Folder Final

Struktur direktori proyek yang benar dan lengkap untuk memenuhi standar pengembangan:

```
final_project/
├── data/
│   ├── Perfumes_dataset.csv       # Dataset mentah asli dari Kaggle
│   ├── perfumes_clean.csv         # Dataset bersih (output preprocessing)
│   └── *.png                      # Gambar visualisasi hasil EDA
├── notebooks/
│   └── eda_preprocessing.ipynb    # Eksplorasi data awal & visualisasi
├── src/
│   ├── preprocessing.py           # Pipeline pembersihan data & feature engineering
│   ├── recommender.py             # Logika model rekomendasi (oleh Hasan)
│   └── app.py                     # Aplikasi web interactive Streamlit (oleh Alif)
├── tests/
│   └── test_recommender.py        # Pengujian sistem rekomendasi dengan pytest (oleh Hasan)
├── requirements.txt               # Daftar dependencies library Python
├── link_aplikasi.txt              # File pengumpulan link (oleh Torikh)
└── README.md                      # Dokumentasi utama proyek
```

---

## 🟢 Phase 1: Project Setup & Environment [COMPLETED]

Langkah awal persiapan repositori dan library yang dibutuhkan untuk pengembangan:

- [x] Buat struktur folder proyek (`data/`, `notebooks/`, `src/`)
- [x] Pindahkan dataset `Perfumes_dataset.csv` ke dalam folder `data/`
- [x] Buat file `requirements.txt` yang memuat semua library utama dan testing:
  ```
  pandas
  scikit-learn
  matplotlib
  seaborn
  streamlit
  numpy
  pytest
  pytest-cov
  ```
- [x] Jalankan instalasi dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🟢 Phase 2: Exploratory Data Analysis (EDA) [COMPLETED]

Eksplorasi data untuk memahami sebaran fitur dan hubungan antar variabel:

- [x] **Load & Inspeksi Data**: Memeriksa dimensi (1.005 baris, 6 kolom), tipe data, dan memastikan tidak ada nilai kosong (*missing values*).
- [x] **Visualisasi Distribusi 4 Kolom Kategorikal**: Membuat plot distribusi fitur `type`, `category`, `target_audience`, dan `longevity`.
- [x] **Visualisasi Top-10 Brand**: Bar chart horizontal yang menampilkan brand dengan jumlah parfum terbanyak.
- [x] **Crosstab Category vs Longevity**: Stacked bar chart yang menganalisis ketahanan wangi berdasarkan kategori aroma.
- [x] **Penyimpanan Asset**: Menyimpan seluruh grafik visualisasi dalam format PNG ke folder `data/` untuk digunakan pada visualisasi README dan Web App.

---

## 🟢 Phase 3: Data Preprocessing & Pipeline [COMPLETED]

Pembersihan data mentah agar siap diproses oleh algoritma machine learning, dikemas dalam script modular `src/preprocessing.py`:

- [x] **Standardisasi Teks**: Mengubah semua nilai teks kolom kategori menjadi *lowercase* dan membuang spasi berlebih (*trim whitespace*).
- [x] **Penanganan Noise & Konsistensi**:
  - Mengubah target audience agar konsisten: `men` ➔ `male`, `women` ➔ `female`.
  - Mengelompokkan variasi ketahanan (*longevity*) menjadi 3 kategori utama: `light`, `medium`, dan `strong`.
  - Mengganti nilai kosong atau string 'nan' dengan label `'unknown'` menggunakan numpy.
- [x] **Deduplikasi Data**: Menghapus baris duplikat berdasarkan kombinasi nama brand dan parfum (`brand` & `perfume`) demi keakuratan rekomendasi.
- [x] **Feature Engineering**: Membuat kolom baru bernama `'combined_features'` yang menggabungkan fitur `type`, `category`, `target_audience`, dan `longevity` dipisahkan dengan spasi (contoh: `"edp woody spicy male strong"`).
- [x] **Ekspor Dataset Bersih**: Menyimpan hasil pembersihan ke file `data/perfumes_clean.csv`.

---

## 🟢 Phase 4: Recommender Engine Development & TDD [COMPLETED]

Pembangunan logika rekomendasi menggunakan metode **Content-Based Filtering** dengan menerapkan pendekatan **Test-Driven Development (TDD)**:

- [x] **Persiapan Pengujian (`tests/test_recommender.py`)** — 30 test cases:
  Hasan membuat file unit test menggunakan `pytest` untuk memverifikasi beberapa skenario sebelum menulis kode model:
  1. **Validasi Load Data** (8 tests): Memastikan fungsi dapat membaca dataset dengan benar, memuat kolom-kolom penting, dan membersihkan anomali residual (header row leak, longevity/audience tidak standar).
  2. **Validasi Matriks Similarity** (5 tests): Memastikan dimensi matriks kemiripan adalah $N \times N$, skor diagonal bernilai `1.0`, seluruh nilai berada di rentang `[0, 1]`, dan matriks simetris.
  3. **Akurasi Rekomendasi** (7 tests): Menguji apakah parfum Woody Spicy & Floriental merekomendasikan aroma serupa dengan skor tinggi, urutan descending, dan jumlah sesuai top_n.
  4. **Pencegahan Self-Recommendation** (2 tests): Memastikan sistem tidak merekomendasikan parfum input itu sendiri untuk 10+ parfum berbeda.
  5. **Penanganan Case-Insensitivity** (3 tests): Menguji lowercase, UPPERCASE, Title Case, dan extra whitespace menghasilkan hasil identik.
  6. **Penanganan Input Tidak Valid** (5 tests): Memastikan system menolak input yang tidak terdaftar, string kosong, spasi, dan nama parsial dengan melempar `ValueError("parfum tidak ditemukan")`.

- [x] **Implementasi Model (`src/recommender.py`)**:
  Hasan mengimplementasikan fungsi-fungsi utama berikut di `src/recommender.py`:
  *   `load_data(filepath)`: Memuat dataset bersih `data/perfumes_clean.csv` + membersihkan 3 anomali residual (header row leak, longevity "6–8 hours", target_audience "gourmand").
  *   `build_similarity_matrix(df)`: Menggunakan `CountVectorizer` untuk mengubah `'combined_features'` menjadi representasi numerik, lalu menghitung matriks kemiripan menggunakan `cosine_similarity`.
  *   `get_recommendations(perfume_name, df, cosine_sim, top_n)`: Mengembalikan DataFrame rekomendasi teratas yang diurutkan descending, mengecualikan parfum input, dan menambahkan skor kemiripan (`similarity_score`).

- [x] **Eksekusi & Verifikasi** — ✅ 30/30 passed, 97% coverage:
  ```
  tests/test_recommender.py  30 passed in 17.21s
  src/recommender.py         97% coverage (39 stmts, 1 miss)
  ```

---

## ⏳ Phase 5: Premium Streamlit Web UI [Alif's Job]

Pembuatan aplikasi antarmuka interaktif yang premium, responsif, dan mudah digunakan:

- [ ] **Arsitektur Antarmuka (`src/app.py`)**:
  Alif membangun UI interaktif menggunakan Streamlit dengan komponen-komponen berikut:
  *   `st.set_page_config()`: Mengatur judul halaman "Fragrance Matchmaker" dan favicon 🧴.
  *   **Dynamic Dropdown Filter**: Menyediakan sidebar filter untuk `Target Audience`, `Kategori Aroma`, dan `Longevity`. Dropdown daftar parfum di halaman utama harus ter-filter secara dinamis agar pengguna tidak menemui *empty state/dead-ends*.
  *   **Interactive Input**: Slider untuk menentukan jumlah rekomendasi (Top 3 sampai 10) dan tombol pencarian.
  *   **Visual Elements (Premium Styling)**:
    *   Mengintegrasikan Custom CSS untuk menghadirkan desain modern (glassmorphism cards, custom font Inter, dark mode aesthetic, dan animasi hover mikro).
    *   Menampilkan metrik parfum pilihan menggunakan metric cards (`st.metric`) secara responsif.
  *   **Hasil Rekomendasi**: Menampilkan tabel rekomendasi yang rapi, lengkap dengan ranking dan similarity score.
- [ ] **Uji Coba Lokal**:
  Menjalankan aplikasi secara lokal untuk memastikan integrasi backend model dan frontend berjalan lancar:
  ```bash
  streamlit run src/app.py
  ```

---

## ⏳ Phase 6: Deployment & Packaging [Torikh & Tim]

Tahap akhir rilis aplikasi dan penyusunan berkas pengumpulan tugas:

- [ ] **Deployment ke Streamlit Cloud**:
  *   Push seluruh perubahan terbaru ke repositori GitHub.
  *   Daftar/login ke [share.streamlit.io](https://share.streamlit.io).
  *   Hubungkan repositori `final_project_DSBeginner`, arahkan file utama ke `src/app.py`, lalu klik **Deploy**.
  *   Salin URL aplikasi publik yang dihasilkan.
- [ ] **Pembuatan Dokumen Pengumpulan**:
  *   **README.md**: Lengkapi deskripsi proyek, cara instalasi, penjelasan algoritma, screenshot UI, dan link aplikasi langsung.
  *   **link_aplikasi.txt**: Buat file teks di root direktori yang berisi URL repositori GitHub dan URL aplikasi Streamlit Cloud yang sudah live.
- [ ] **Final Code Push & Submit**:
  *   Melakukan commit akhir dan mem-push seluruh file ke GitHub.
  *   Mengumpulkan URL repositori GitHub dan link web app ke tempat pengumpulan tugas Study Club.

---

## 📅 Timeline Rencana Kerja

| Tanggal | Target Output | Status | PIC |
|---------|---------------|--------|-----|
| **15-16 Mei** | Setup proyek, import dataset, & penyusunan notebook EDA | **🟢 SELESAI** | Torikh |
| **17-18 Mei** | Modularisasi preprocessing script & pembuatan `perfumes_clean.csv` | **🟢 SELESAI** | Torikh |
| **19-20 Mei** | Penulisan Unit Test (`tests/`) & Coding Model Rekomendasi (`recommender.py`) | **🟢 SELESAI** | Hasan |
| **20-21 Mei** | Coding Web App Streamlit (`app.py`) & Premium Glassmorphism styling | **⏳ PENDING** | Alif |
| **22 Mei** | Integrasi Model-UI, Manual Testing, & Deployment Streamlit Cloud | **⏳ PENDING** | Tim |
| **23 Mei** | Penyusunan `README.md` & `link_aplikasi.txt` (Deadline 23:59 WIB) | **⏳ PENDING** | Torikh |
