# Fragrance Matchmaker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membangun sistem rekomendasi parfum berbasis Content-Based Filtering yang di-deploy sebagai aplikasi Streamlit interaktif.

**Architecture:** User memilih 1 parfum favorit → sistem menghitung Cosine Similarity terhadap seluruh parfum di dataset → menampilkan Top-N parfum paling mirip berdasarkan fitur `type`, `category`, `target_audience`, dan `longevity`.

**Tech Stack:** Python, Pandas, Scikit-learn (CountVectorizer, cosine_similarity), Matplotlib/Seaborn, Streamlit

**Deadline:** 23 Mei 2026 | **Team:** Torikh Abdullah Naser & Alif Ilham Rhamadan

**Dataset:** [Perfume Dataset — Kaggle (Ayush)](https://www.kaggle.com/datasets/ayushghawana/perfume-dataset) — 1.005 rows, 6 columns, 0 missing values.

---

## Struktur Folder Final

```
final_project/
├── data/
│   └── Perfumes_dataset.csv
├── notebooks/
│   └── eda_preprocessing.ipynb
├── src/
│   ├── recommender.py
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Phase 1: Project Setup

- [ ] Buat struktur folder (`data/`, `notebooks/`, `src/`)
- [ ] Copy `Perfumes_dataset.csv` dari `Tugas2/archive2/` ke `final_project/data/`
- [ ] Buat `requirements.txt`: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`
- [ ] `pip install -r requirements.txt`

---

## Phase 2: EDA & Data Visualization

> File: `notebooks/eda_preprocessing.ipynb`

- [ ] **Load & inspeksi data**

```python
import pandas as pd
df = pd.read_csv('../data/Perfumes_dataset.csv')
print(df.shape)            # (1005, 6)
print(df.info())
print(df.isnull().sum())   # Harus 0 semua
print(df.head(10))
```

- [ ] **Visualisasi distribusi 4 kolom kategorikal** (type, category, target_audience, longevity)

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['type'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Distribusi Tipe Parfum')

df['category'].value_counts().plot(kind='barh', ax=axes[0,1], color='salmon')
axes[0,1].set_title('Distribusi Kategori Aroma')

df['target_audience'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
axes[1,0].set_title('Target Audience')

df['longevity'].value_counts().plot(kind='bar', ax=axes[1,1], color='mediumseagreen')
axes[1,1].set_title('Distribusi Longevity')

plt.tight_layout()
plt.savefig('../data/eda_distributions.png', dpi=150)
plt.show()
```

- [ ] **Top-10 Brand** (bar chart horizontal)
- [ ] **Crosstab Category vs Longevity** (stacked bar chart)

---

## Phase 3: Data Preprocessing

> File: Lanjutan di `notebooks/eda_preprocessing.ipynb`

- [ ] **Standardisasi semua kolom teks ke lowercase**

```python
text_cols = ['brand', 'perfume', 'type', 'category', 'target_audience', 'longevity']
for col in text_cols:
    df[col] = df[col].str.strip().str.lower()
```

- [ ] **Cek & hapus duplikat**

```python
dupes = df.duplicated(subset=['brand', 'perfume']).sum()
print(f'Duplikat: {dupes}')
if dupes > 0:
    df = df.drop_duplicates(subset=['brand', 'perfume']).reset_index(drop=True)
```

- [ ] **Buat kolom `combined_features`** (gabungan fitur untuk vektorisasi)

```python
df['combined_features'] = (
    df['type'] + ' ' +
    df['category'] + ' ' +
    df['target_audience'] + ' ' +
    df['longevity']
)
# Contoh output: "edp fresh scent male strong"
```

- [ ] **Simpan data bersih** ke `data/perfumes_clean.csv`

---

## Phase 4: Feature Engineering & Similarity Matrix

> File: `src/recommender.py`

- [ ] **Fungsi `load_data()`** — Load CSV yang sudah bersih
- [ ] **Fungsi `build_similarity_matrix()`** — CountVectorizer + Cosine Similarity

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(df):
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(feature_matrix)
    return cosine_sim
```

> **Kenapa CountVectorizer, bukan TF-IDF?** Data kita sudah berupa kata-kata kategori pendek dan bersih. TF-IDF lebih cocok untuk teks panjang (review/deskripsi).

- [ ] **Fungsi `get_recommendations()`** — Input nama parfum, output Top-N DataFrame + similarity score

```python
def get_recommendations(perfume_name, df, cosine_sim, top_n=5):
    idx = df[df['perfume'] == perfume_name.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Skip diri sendiri

    perfume_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 4) for i in sim_scores]

    result = df.iloc[perfume_indices][
        ['brand', 'perfume', 'type', 'category', 'target_audience', 'longevity']
    ].copy()
    result['similarity_score'] = scores
    result = result.reset_index(drop=True)
    result.index += 1  # Ranking mulai dari 1
    return result
```

- [ ] **Test di notebook** — Cari rekomendasi untuk "club de nuit intense man", pastikan hasilnya masuk akal (Woody Spicy lainnya muncul)

---

## Phase 5: Streamlit Deployment

> File: `src/app.py`

- [ ] **Buat UI Streamlit** dengan komponen:
  - `st.set_page_config()` — judul "Fragrance Matchmaker", icon 🧴
  - **Sidebar filters:** `st.selectbox` untuk Target Audience, Kategori Aroma, Longevity
  - **Main area:** `st.selectbox` untuk pilih parfum (filtered)
  - `st.slider` untuk jumlah rekomendasi (3-10)
  - `st.metric` x4 menampilkan info parfum yang dipilih
  - `st.button` "Cari Rekomendasi" → `st.dataframe` hasil Top-N
  - Footer credit tim

- [ ] **Test lokal:** `streamlit run src/app.py`
  - Pastikan filter bekerja
  - Pastikan hasil rekomendasi terurut descending by similarity score
  - Pastikan parfum Woody Spicy merekomendasikan Woody Spicy lainnya

- [ ] **Deploy ke Streamlit Cloud:**
  1. Push ke GitHub
  2. Buka [share.streamlit.io](https://share.streamlit.io)
  3. Connect repo → pilih `src/app.py`
  4. Deploy → salin link

---

## Phase 6: Packaging & Pengumpulan

- [ ] Buat `README.md` (judul, tim, dataset link, cara run, link app)
- [ ] Buat `link_app.txt` berisi URL Streamlit Cloud
- [ ] Siapkan folder: `FP_Torikh[NIM]_Alif[NIM]_Hasan[NIM]`
- [ ] Upload ke tempat pengumpulan

---

## Verification Plan

### Automated
```bash
python -c "
from src.recommender import load_data, build_similarity_matrix, get_recommendations
df = load_data('data/perfumes_clean.csv')
sim = build_similarity_matrix(df)
result = get_recommendations('club de nuit intense man', df, sim, 5)
assert len(result) == 5
assert 'similarity_score' in result.columns
print('ALL TESTS PASSED')
"
```

### Manual
- Parfum Woody Spicy → rekomendasi mayoritas Woody Spicy ✓
- Similarity score antara 0-1, terurut descending ✓
- Filter sidebar mengubah daftar parfum yang tersedia ✓

---

## Timeline

| Tanggal | Target |
|---------|--------|
| 15-16 Mei | Phase 1-2: Setup + EDA |
| 17-18 Mei | Phase 3-4: Preprocessing + Similarity Matrix |
| 19-20 Mei | Phase 5: Streamlit App |
| 21-22 Mei | Phase 6: Deploy + Packaging |
| **23 Mei** | **DEADLINE** |
