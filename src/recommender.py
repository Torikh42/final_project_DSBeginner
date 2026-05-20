"""
recommender.py — Content-Based Filtering Recommender Engine
Modul ini berisi logika utama sistem rekomendasi parfum menggunakan
metode Content-Based Filtering dengan CountVectorizer & Cosine Similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Fungsi Utama ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset bersih dari file CSV dan melakukan validasi + 
    pembersihan anomali residual yang lolos dari tahap preprocessing.

    Anomali yang ditangani:
    1. Header row leak (baris dimana brand == 'brand')
    2. Longevity tidak standar ('6–8 hours') → dimap ke 'medium'
    3. Target audience tidak standar ('gourmand') → dimap ke 'unisex'
    4. Rebuild combined_features setelah perbaikan

    Args:
        filepath (str): Path ke file CSV dataset bersih.

    Returns:
        pd.DataFrame: Dataset yang sudah divalidasi dan dibersihkan.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika kolom wajib tidak ada di dataset.
    """
    df = pd.read_csv(filepath)

    # --- Validasi kolom wajib ---
    required_cols = [
        'brand', 'perfume', 'type', 'category',
        'target_audience', 'longevity', 'combined_features'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    # --- Pembersihan anomali residual ---

    # 1. Hapus header row yang bocor ke data (brand == 'brand')
    df = df[df['brand'] != 'brand'].reset_index(drop=True)

    # 2. Fix longevity tidak standar → 'medium'
    #    (6–8 hours setara medium dalam standar industri parfum)
    valid_longevity = {'light', 'medium', 'strong'}
    mask_lon = ~df['longevity'].isin(valid_longevity)
    df.loc[mask_lon, 'longevity'] = 'medium'

    # 3. Fix target_audience tidak standar → 'unisex'
    #    ('gourmand' adalah kategori aroma, bukan audience)
    valid_audience = {'male', 'female', 'unisex'}
    mask_aud = ~df['target_audience'].isin(valid_audience)
    df.loc[mask_aud, 'target_audience'] = 'unisex'

    # 4. Rebuild combined_features setelah perbaikan data
    df['combined_features'] = (
        df['type'] + ' ' +
        df['category'] + ' ' +
        df['target_audience'] + ' ' +
        df['longevity']
    )

    return df


def build_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Membangun matriks cosine similarity dari kolom combined_features.

    Menggunakan CountVectorizer (bukan TF-IDF) karena fitur berupa
    kata-kata kategori pendek dan bersih (contoh: "edp woody spicy male strong"),
    sehingga pembobotan term frequency tidak diperlukan.

    Proses:
    1. CountVectorizer mengubah teks combined_features → matriks fitur numerik
    2. Cosine similarity menghitung kemiripan antar semua pasangan parfum

    Args:
        df (pd.DataFrame): Dataset dengan kolom 'combined_features'.

    Returns:
        np.ndarray: Matriks similarity berukuran N × N, dimana N = jumlah parfum.
                    Nilai berkisar antara 0.0 (tidak mirip) hingga 1.0 (identik).
    """
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(feature_matrix)
    return cosine_sim


def get_recommendations(perfume_name: str, df: pd.DataFrame,
                        cosine_sim: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """
    Mengembalikan Top-N rekomendasi parfum berdasarkan cosine similarity.

    Fitur utama:
    - Case-insensitive search (pencarian tidak sensitif huruf besar/kecil)
    - Self-recommendation prevention (parfum input tidak muncul di hasil)
    - Hasil diurutkan descending berdasarkan similarity score

    Args:
        perfume_name (str): Nama parfum input dari user (case-insensitive).
        df (pd.DataFrame): Dataset parfum yang sudah dibersihkan.
        cosine_sim (np.ndarray): Matriks cosine similarity.
        top_n (int): Jumlah rekomendasi yang dikembalikan (default: 5).

    Returns:
        pd.DataFrame: DataFrame berisi top_n rekomendasi dengan kolom:
            brand, perfume, type, category, target_audience, longevity,
            similarity_score

    Raises:
        ValueError: Jika parfum tidak ditemukan dalam dataset.
                    Pesan error: "parfum tidak ditemukan"
    """
    # Case-insensitive search
    perfume_lower = perfume_name.strip().lower()
    matches = df[df['perfume'].str.lower() == perfume_lower]

    if matches.empty:
        raise ValueError("parfum tidak ditemukan")

    # Ambil index pertama jika ada nama duplikat
    idx = matches.index[0]

    # Ambil skor similarity untuk parfum terpilih
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Urutkan descending berdasarkan skor similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Hapus self-recommendation (parfum input itu sendiri)
    sim_scores = [(i, score) for i, score in sim_scores if i != idx]

    # Ambil top_n rekomendasi
    top_scores = sim_scores[:top_n]

    # Buat DataFrame hasil rekomendasi
    top_indices = [i for i, _ in top_scores]
    result = df.iloc[top_indices][[
        'brand', 'perfume', 'type', 'category',
        'target_audience', 'longevity'
    ]].copy()
    result['similarity_score'] = [score for _, score in top_scores]
    result = result.reset_index(drop=True)

    return result
