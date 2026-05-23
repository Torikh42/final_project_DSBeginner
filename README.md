# 🧴 Fragrance Matchmaker
### Sistem Rekomendasi Parfum Berbasis Content-Based Filtering

> Final Project — Data Science Study Club Beginner 2026

---

## 👥 Tim

| Nama |
|------|
| Torikh Abdullah Naser | 
| Alif Ilham Rhamadan   |
| Hasan Shofiyyurrahman |


---

## 📌 Deskripsi Proyek

**Fragrance Matchmaker** adalah sistem rekomendasi parfum yang membantu pengguna menemukan parfum dengan karakteristik serupa berdasarkan parfum favorit yang mereka pilih.

Sistem ini menggunakan metode **Content-Based Filtering** — sebuah pendekatan machine learning yang merekomendasikan item berdasarkan kemiripan atribut/fitur antar produk, bukan berdasarkan data interaksi pengguna lain (*collaborative filtering*).

### Alur Kerja Sistem

```
User pilih 1 parfum favorit
        ↓
Filter by: Audience / Category / Longevity (opsional)
        ↓
Sistem hitung Cosine Similarity
terhadap seluruh parfum di dataset
        ↓
Tampilkan Top-N rekomendasi
beserta Similarity Score (0.0 – 1.0)
```

---

## 📊 Dataset

| Atribut | Detail |
|---------|--------|
| **Nama** | Perfume Dataset |
| **Sumber** | [Kaggle — Ayush Ghawana](https://www.kaggle.com/datasets/ayushghawana/perfume-dataset) |
| **Lisensi** | CC BY 4.0 (Attribution 4.0 International) |
| **Jumlah Data** | 1.004 baris |
| **Jumlah Kolom** | 6 kolom |
| **Missing Values** | 0 (tidak ada) |

### Deskripsi Kolom

| Kolom | Deskripsi | Contoh |
|-------|-----------|--------|
| `brand` | Merk/perusahaan pembuat parfum | Armaf, Dior |
| `perfume` | Nama spesifik parfum | Club de Nuit Intense |
| `type` | Konsentrasi parfum | EDP, EDT, Parfum |
| `category` | Keluarga/kategori aroma | Woody Spicy, Fresh Scent, Floral Fruity |
| `target_audience` | Target pengguna | Male, Female, Unisex |
| `longevity` | Ketahanan wangi di kulit | Strong, Medium |

---

## 🛠️ Tech Stack

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.x | Bahasa Pemrograman Utama |
| Pandas | ≥2.0 | Manipulasi & Analisis Data |
| Scikit-learn | ≥1.3 | CountVectorizer, Cosine Similarity |
| Matplotlib | ≥3.7 | Visualisasi Data (EDA) |
| Seaborn | ≥0.12 | Visualisasi Data (EDA) |
| Streamlit | ≥1.30 | Deployment Aplikasi Interaktif |

---

## 📁 Struktur Proyek

```
final_project/
├── data/
│   ├── Perfumes_dataset.csv       # Dataset asli dari Kaggle
│   └── perfumes_clean.csv         # Dataset setelah preprocessing
├── notebooks/
│   └── eda_preprocessing.ipynb    # Eksplorasi data & preprocessing
├── src/
│   ├── recommender.py             # Logika model rekomendasi
│   └── app.py                     # Aplikasi Streamlit
├── requirements.txt               # Daftar dependencies
└── README.md                      # File ini
```

---

## ⚙️ Cara Menjalankan

### 1. Clone / Unduh Proyek
```bash
git clone <repo-url>
cd final_project
```
Atau unduh folder ini secara manual.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run src/app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 🤖 Cara Kerja Model

### 1. Feature Engineering
Keempat kolom fitur digabungkan menjadi satu string `combined_features`:
```
combined_features = type + category + target_audience + longevity
# Contoh: "edp woody spicy male strong"
```

### 2. Vektorisasi (CountVectorizer)
String `combined_features` diubah menjadi representasi numerik (matriks fitur) menggunakan `CountVectorizer` dari Scikit-learn:
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])
```

> **Mengapa CountVectorizer, bukan TF-IDF?**
> Data kita berupa kata-kata kategori pendek dan bersih (misal: *"edp woody spicy male strong"*). TF-IDF lebih optimal untuk teks panjang seperti review atau deskripsi produk.

### 3. Cosine Similarity
Matriks kemiripan dihitung menggunakan **Cosine Similarity**. Nilainya antara 0.0 – 1.0, dimana 1.0 berarti identik sempurna:
```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(feature_matrix)
```

### 4. Rekomendasi
Sistem mengambil baris yang sesuai dengan parfum pilihan user di matriks similarity, mengurutkan skor secara descending, dan mengembalikan Top-N parfum teratas.

---

## 🎯 Fitur Aplikasi

- 🔍 **Filter Sidebar** — Filter daftar parfum berdasarkan Target Audience, Kategori Aroma, dan Longevity sebelum memilih
- 📋 **Pilih Parfum** — Dropdown berisi parfum yang sesuai dengan filter
- 📊 **Info Parfum** — Menampilkan atribut parfum yang dipilih (Brand, Type, Category, Longevity)
- 🎚️ **Slider Top-N** — Pilih jumlah rekomendasi yang ditampilkan (3 hingga 10)
- 🏆 **Tabel Rekomendasi** — Menampilkan parfum rekomendasi dengan Similarity Score, diurutkan dari yang paling mirip

---

## 🔗 Link Aplikasi

> 🚀 **[https://fragrance-matchmaker.streamlit.app/](https://fragrance-matchmaker.streamlit.app/)**

---

## 📚 Referensi

- Kaggle Dataset: https://www.kaggle.com/datasets/ayushghawana/perfume-dataset
- Scikit-learn CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- Scikit-learn Cosine Similarity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- Streamlit Documentation: https://docs.streamlit.io

---

*Study Club Data Science Beginner — 2026*
