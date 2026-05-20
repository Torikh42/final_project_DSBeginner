"""
test_recommender.py — Unit Test Suite untuk Recommender Engine
Pengujian komprehensif menggunakan pytest untuk memverifikasi
akurasi, robustness, dan keandalan sistem rekomendasi parfum.

Skenario pengujian:
    1. Validasi Load Data
    2. Validasi Matriks Similarity
    3. Akurasi Rekomendasi
    4. Pencegahan Self-Recommendation
    5. Penanganan Case-Insensitivity
    6. Penanganan Input Tidak Valid
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import load_data, build_similarity_matrix, get_recommendations

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'perfumes_clean.csv')


@pytest.fixture(scope="module")
def df():
    """Load dataset sekali untuk seluruh test dalam modul ini."""
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def cosine_sim(df):
    """Build similarity matrix sekali untuk seluruh test dalam modul ini."""
    return build_similarity_matrix(df)


# ─── Test 1: Validasi Load Data ───────────────────────────────────────────────

class TestLoadData:
    """Memastikan fungsi load_data() dapat membaca dataset dengan benar
    dan memuat kolom-kolom penting."""

    def test_load_returns_dataframe(self, df):
        """Data yang dimuat harus berupa pandas DataFrame."""
        assert isinstance(df, pd.DataFrame)

    def test_load_has_required_columns(self, df):
        """Dataset harus memiliki semua 7 kolom penting."""
        required = [
            'brand', 'perfume', 'type', 'category',
            'target_audience', 'longevity', 'combined_features'
        ]
        for col in required:
            assert col in df.columns, f"Kolom '{col}' tidak ditemukan"

    def test_load_not_empty(self, df):
        """Dataset tidak boleh kosong setelah dimuat."""
        assert len(df) > 0

    def test_no_header_row_leak(self, df):
        """Tidak boleh ada baris header yang bocor ke data."""
        assert not (df['brand'] == 'brand').any(), \
            "Ditemukan header row leak di dataset"

    def test_no_null_in_key_columns(self, df):
        """Tidak boleh ada missing values di kolom kunci."""
        key_cols = ['brand', 'perfume', 'combined_features']
        null_counts = df[key_cols].isnull().sum()
        assert null_counts.sum() == 0, \
            f"Ditemukan null values: {null_counts[null_counts > 0].to_dict()}"

    def test_clean_longevity_values(self, df):
        """Kolom longevity hanya boleh berisi: light, medium, atau strong."""
        valid = {'light', 'medium', 'strong'}
        actual = set(df['longevity'].unique())
        invalid = actual - valid
        assert actual.issubset(valid), \
            f"Longevity memiliki nilai tidak valid: {invalid}"

    def test_clean_target_audience_values(self, df):
        """Kolom target_audience hanya boleh berisi: male, female, atau unisex."""
        valid = {'male', 'female', 'unisex'}
        actual = set(df['target_audience'].unique())
        invalid = actual - valid
        assert actual.issubset(valid), \
            f"Target audience memiliki nilai tidak valid: {invalid}"

    def test_combined_features_not_empty(self, df):
        """Kolom combined_features tidak boleh berisi string kosong."""
        empty_count = (df['combined_features'].str.strip() == '').sum()
        assert empty_count == 0, \
            f"Ditemukan {empty_count} baris dengan combined_features kosong"


# ─── Test 2: Validasi Matriks Similarity ──────────────────────────────────────

class TestSimilarityMatrix:
    """Memastikan dimensi matriks kemiripan adalah N×N, skor diagonal
    bernilai 1.0, dan seluruh nilai berada di rentang [0, 1]."""

    def test_matrix_shape_is_square(self, df, cosine_sim):
        """Matriks similarity harus berukuran N × N."""
        n = len(df)
        assert cosine_sim.shape == (n, n), \
            f"Expected ({n}, {n}), got {cosine_sim.shape}"

    def test_diagonal_is_one(self, cosine_sim):
        """Diagonal matriks harus bernilai 1.0 (self-similarity sempurna)."""
        diagonal = np.diag(cosine_sim)
        np.testing.assert_array_almost_equal(
            diagonal, np.ones(len(diagonal)),
            decimal=6,
            err_msg="Diagonal matriks tidak bernilai 1.0"
        )

    def test_all_values_in_valid_range(self, cosine_sim):
        """Semua nilai similarity harus berada di rentang [0, 1]."""
        assert cosine_sim.min() >= -1e-9, \
            f"Nilai minimum di bawah 0: {cosine_sim.min()}"
        assert cosine_sim.max() <= 1.0 + 1e-9, \
            f"Nilai maximum di atas 1: {cosine_sim.max()}"

    def test_matrix_is_symmetric(self, cosine_sim):
        """Matriks similarity harus simetris: sim(A,B) == sim(B,A)."""
        np.testing.assert_array_almost_equal(
            cosine_sim, cosine_sim.T,
            decimal=6,
            err_msg="Matriks similarity tidak simetris"
        )

    def test_matrix_dtype_is_float(self, cosine_sim):
        """Matriks similarity harus bertipe float."""
        assert np.issubdtype(cosine_sim.dtype, np.floating), \
            f"Expected float dtype, got {cosine_sim.dtype}"


# ─── Test 3: Akurasi Rekomendasi ──────────────────────────────────────────────

class TestRecommendationAccuracy:
    """Menguji apakah parfum dengan kategori tertentu merekomendasikan
    aroma serupa dengan skor tinggi."""

    def test_woody_spicy_recommends_similar_category(self, df, cosine_sim):
        """Parfum Woody Spicy harus merekomendasikan mayoritas aroma serupa."""
        # Ambil parfum pertama dengan kategori 'woody spicy'
        woody_perfumes = df[df['category'] == 'woody spicy']
        perfume_name = woody_perfumes.iloc[0]['perfume']

        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=5)

        # Minimal 3 dari 5 rekomendasi harus berkategori 'woody spicy'
        woody_count = recs[recs['category'] == 'woody spicy'].shape[0]
        assert woody_count >= 3, \
            f"Hanya {woody_count}/5 rekomendasi yang woody spicy"

    def test_floriental_recommends_similar_category(self, df, cosine_sim):
        """Parfum Floriental harus merekomendasikan mayoritas aroma serupa."""
        floriental_perfumes = df[df['category'] == 'floriental']
        perfume_name = floriental_perfumes.iloc[0]['perfume']

        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=5)

        floriental_count = recs[recs['category'] == 'floriental'].shape[0]
        assert floriental_count >= 3, \
            f"Hanya {floriental_count}/5 rekomendasi yang floriental"

    def test_top_recommendation_has_high_score(self, df, cosine_sim):
        """Rekomendasi teratas harus memiliki skor similarity tinggi (>= 0.5)."""
        perfume_name = df.iloc[0]['perfume']
        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=3)

        top_score = recs.iloc[0]['similarity_score']
        assert top_score >= 0.5, \
            f"Skor rekomendasi teratas terlalu rendah: {top_score:.4f}"

    def test_recommendations_sorted_descending(self, df, cosine_sim):
        """Rekomendasi harus diurutkan dari skor tertinggi ke terendah."""
        perfume_name = df.iloc[0]['perfume']
        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=5)

        scores = recs['similarity_score'].tolist()
        assert scores == sorted(scores, reverse=True), \
            "Rekomendasi tidak diurutkan descending"

    def test_correct_number_of_recommendations(self, df, cosine_sim):
        """Jumlah rekomendasi harus sesuai dengan parameter top_n."""
        perfume_name = df.iloc[0]['perfume']
        for n in [3, 5, 10]:
            recs = get_recommendations(perfume_name, df, cosine_sim, top_n=n)
            assert len(recs) == n, \
                f"Expected {n} rekomendasi, got {len(recs)}"

    def test_result_has_required_columns(self, df, cosine_sim):
        """Hasil rekomendasi harus memiliki kolom yang diperlukan."""
        perfume_name = df.iloc[0]['perfume']
        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=3)

        expected_cols = [
            'brand', 'perfume', 'type', 'category',
            'target_audience', 'longevity', 'similarity_score'
        ]
        for col in expected_cols:
            assert col in recs.columns, \
                f"Kolom '{col}' tidak ada di hasil rekomendasi"

    def test_similarity_score_in_valid_range(self, df, cosine_sim):
        """Semua similarity score di hasil rekomendasi harus antara 0 dan 1."""
        perfume_name = df.iloc[0]['perfume']
        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=10)

        assert recs['similarity_score'].min() >= 0.0, \
            "Ada similarity score di bawah 0"
        assert recs['similarity_score'].max() <= 1.0, \
            "Ada similarity score di atas 1"


# ─── Test 4: Pencegahan Self-Recommendation ───────────────────────────────────

class TestSelfRecommendation:
    """Memastikan sistem tidak merekomendasikan parfum input itu sendiri."""

    def test_no_self_recommendation(self, df, cosine_sim):
        """Parfum input tidak boleh muncul di daftar rekomendasi."""
        perfume_name = df.iloc[0]['perfume']
        recs = get_recommendations(perfume_name, df, cosine_sim, top_n=10)

        assert perfume_name not in recs['perfume'].values, \
            f"Self-recommendation detected: '{perfume_name}' muncul di hasil"

    def test_no_self_recommendation_multiple_perfumes(self, df, cosine_sim):
        """Pengujian self-recommendation untuk beberapa parfum berbeda."""
        # Uji 10 parfum pertama
        for i in range(min(10, len(df))):
            perfume_name = df.iloc[i]['perfume']
            recs = get_recommendations(perfume_name, df, cosine_sim, top_n=5)
            assert perfume_name not in recs['perfume'].values, \
                f"Self-recommendation detected pada index {i}: '{perfume_name}'"


# ─── Test 5: Penanganan Case-Insensitivity ────────────────────────────────────

class TestCaseInsensitivity:
    """Menguji apakah pencarian parfum bersifat case-insensitive."""

    def test_lowercase_equals_uppercase(self, df, cosine_sim):
        """Pencarian huruf kecil dan huruf besar harus menghasilkan rekomendasi sama."""
        original = df.iloc[0]['perfume']  # Sudah lowercase di dataset

        recs_lower = get_recommendations(original.lower(), df, cosine_sim, top_n=5)
        recs_upper = get_recommendations(original.upper(), df, cosine_sim, top_n=5)

        pd.testing.assert_frame_equal(
            recs_lower, recs_upper,
            obj="Hasil lowercase vs uppercase"
        )

    def test_mixed_case_equals_lowercase(self, df, cosine_sim):
        """Pencarian mixed case (Title Case) harus sama dengan lowercase."""
        original = df.iloc[0]['perfume']

        recs_lower = get_recommendations(original.lower(), df, cosine_sim, top_n=5)
        recs_title = get_recommendations(original.title(), df, cosine_sim, top_n=5)

        pd.testing.assert_frame_equal(
            recs_lower, recs_title,
            obj="Hasil lowercase vs Title Case"
        )

    def test_with_extra_whitespace(self, df, cosine_sim):
        """Pencarian dengan spasi tambahan harus tetap berfungsi."""
        original = df.iloc[0]['perfume']

        recs_clean = get_recommendations(original, df, cosine_sim, top_n=5)
        recs_spaced = get_recommendations(f"  {original}  ", df, cosine_sim, top_n=5)

        pd.testing.assert_frame_equal(
            recs_clean, recs_spaced,
            obj="Hasil clean vs extra whitespace"
        )


# ─── Test 6: Penanganan Input Tidak Valid ─────────────────────────────────────

class TestInvalidInput:
    """Memastikan sistem menolak input yang tidak terdaftar dengan
    melempar ValueError('parfum tidak ditemukan')."""

    def test_nonexistent_perfume_raises_error(self, df, cosine_sim):
        """Input parfum yang tidak ada harus melempar ValueError."""
        with pytest.raises(ValueError, match="parfum tidak ditemukan"):
            get_recommendations("parfum_xyz_tidak_ada_12345", df, cosine_sim)

    def test_empty_string_raises_error(self, df, cosine_sim):
        """Input string kosong harus melempar ValueError."""
        with pytest.raises(ValueError, match="parfum tidak ditemukan"):
            get_recommendations("", df, cosine_sim)

    def test_whitespace_only_raises_error(self, df, cosine_sim):
        """Input hanya spasi harus melempar ValueError."""
        with pytest.raises(ValueError, match="parfum tidak ditemukan"):
            get_recommendations("   ", df, cosine_sim)

    def test_random_gibberish_raises_error(self, df, cosine_sim):
        """Input acak/random harus melempar ValueError."""
        with pytest.raises(ValueError, match="parfum tidak ditemukan"):
            get_recommendations("asdf!@#$%^&*()", df, cosine_sim)

    def test_partial_name_raises_error(self, df, cosine_sim):
        """Input nama parsial (tidak lengkap) harus melempar ValueError
        karena sistem menggunakan exact match."""
        # Ambil parfum yang ada, tapi hanya gunakan sebagian namanya
        full_name = df.iloc[0]['perfume']
        partial = full_name[:3]  # Hanya 3 karakter pertama

        # Hanya raise error jika partial bukan nama parfum yang valid
        if partial not in df['perfume'].str.lower().values:
            with pytest.raises(ValueError, match="parfum tidak ditemukan"):
                get_recommendations(partial, df, cosine_sim)
