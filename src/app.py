"""
app.py — Premium Streamlit Web UI for Fragrance Matchmaker
Designed by Alif Ilham Rhamadan (Phase 5)
Design System: CreateSpace — Glassmorphism · Poppins · DM Sans · #E11D48
"""

import os
import sys
import streamlit as st
import pandas as pd

# ─── Path resolution: works both from project root and from src/ ───────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

from recommender import load_data, build_similarity_matrix, get_recommendations

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "perfumes_clean.csv")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fragrance Matchmaker",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CreateSpace Design System CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&family=Fira+Code:wght@400&display=swap');

/* ── CSS Variables (CreateSpace tokens) ── */
:root {
    --color-primary:   #E11D48;
    --color-secondary: #2563EB;
    --color-tertiary:  #FACC15;
    --color-success:   #16A34A;
    --color-warning:   #D97706;
    --color-error:     #DC2626;
    --color-info:      #2563EB;
    --surface-base:    #0F0F14;
    --surface-card:    rgba(255,255,255,0.06);
    --surface-glass:   rgba(255,255,255,0.10);
    --border-glass:    rgba(255,255,255,0.15);
    --text-primary:    #F9FAFB;
    --text-secondary:  #9CA3AF;
    --text-muted:      #6B7280;
    --shadow-glass: 0 8px 32px rgba(0,0,0,0.40);
    --shadow-md:    0 4px 16px rgba(0,0,0,0.30);
    --shadow-color: 0 8px 24px rgba(225,29,72,0.35);
}

/* ── Global reset & dark mode ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--surface-base) !important;
    color: var(--text-primary) !important;
}

/* ── Streamlit main container ── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1280px;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 20, 0.95) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.25rem !important;
}

/* ── Sidebar labels & text ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Selectbox & input fields ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--surface-glass) !important;
    border: 1.5px solid var(--border-glass) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    backdrop-filter: blur(16px);
    transition: border-color 150ms ease;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(255,255,255,0.3) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--color-secondary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.35) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--color-primary) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--color-primary) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--color-primary) 0%, #be123c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.8rem !important;
    width: 100% !important;
    height: 48px !important;
    cursor: pointer;
    transition: all 150ms ease !important;
    box-shadow: var(--shadow-color) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(225,29,72,0.50) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ─────────────────────────────────────────────────────────────────────────────
   HERO HEADER
───────────────────────────────────────────────────────────────────────────── */
.hero-wrapper {
    background: linear-gradient(135deg,
        rgba(225,29,72,0.18) 0%,
        rgba(37,99,235,0.12) 60%,
        rgba(250,204,21,0.08) 100%);
    border: 1px solid var(--border-glass);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-glass);
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(225,29,72,0.22) 0%, transparent 70%);
    top: -120px; right: -80px;
    pointer-events: none;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(37,99,235,0.18) 0%, transparent 70%);
    bottom: -100px; left: -60px;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(225,29,72,0.15);
    border: 1px solid rgba(225,29,72,0.35);
    border-radius: 9999px;
    padding: 4px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #fb7185;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #f9fafb 0%, #e11d48 60%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem 0;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    font-weight: 400;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 640px;
    margin: 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
   METRIC CARDS — Selected Perfume Info
───────────────────────────────────────────────────────────────────────────── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--surface-glass);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-glass);
    transition: transform 200ms ease, box-shadow 200ms ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    border-radius: 16px 16px 0 0;
}
.metric-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Poppins', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    text-transform: capitalize;
}
.metric-icon {
    font-size: 22px;
    margin-bottom: 8px;
    display: block;
}

/* ─────────────────────────────────────────────────────────────────────────────
   SECTION HEADER
───────────────────────────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-glass);
}
.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-count {
    background: rgba(225,29,72,0.15);
    border: 1px solid rgba(225,29,72,0.3);
    border-radius: 9999px;
    padding: 2px 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #fb7185;
}

/* ─────────────────────────────────────────────────────────────────────────────
   RECOMMENDATION CARDS
───────────────────────────────────────────────────────────────────────────── */
.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 2.5rem;
}
.rec-card {
    background: var(--surface-card);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-glass);
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
    position: relative;
    overflow: hidden;
}
.rec-card:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-md);
    border-color: rgba(225,29,72,0.4);
}
.rec-rank {
    position: absolute;
    top: 16px; right: 16px;
    background: linear-gradient(135deg, var(--color-primary), #be123c);
    color: #fff;
    font-family: 'Poppins', sans-serif;
    font-size: 12px;
    font-weight: 700;
    border-radius: 9999px;
    padding: 3px 10px;
    letter-spacing: 0.04em;
}
.rec-score-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 9999px;
    height: 5px;
    margin-bottom: 12px;
    overflow: hidden;
}
.rec-score-bar {
    height: 5px;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
    transition: width 0.6s ease;
}
.rec-perfume {
    font-family: 'Poppins', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    line-height: 1.3;
    padding-right: 48px; /* space for rank badge */
}
.rec-brand {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--color-primary);
    margin: 0 0 16px 0;
    opacity: 0.85;
}
.rec-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 14px;
}
.rec-tag {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 9999px;
    padding: 3px 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: capitalize;
}
.rec-score-label {
    font-family: 'Fira Code', monospace;
    font-size: 12px;
    color: var(--text-muted);
}
.rec-score-val {
    font-family: 'Fira Code', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--color-tertiary);
}

/* ─────────────────────────────────────────────────────────────────────────────
   FULL TABLE VIEW
───────────────────────────────────────────────────────────────────────────── */
.table-wrapper {
    background: var(--surface-glass);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-glass);
    margin-bottom: 2rem;
}

/* ─────────────────────────────────────────────────────────────────────────────
   EMPTY / WELCOME STATE
───────────────────────────────────────────────────────────────────────────── */
.welcome-box {
    background: var(--surface-glass);
    border: 1px dashed var(--border-glass);
    border-radius: 24px;
    padding: 4rem 2rem;
    text-align: center;
    backdrop-filter: blur(16px);
}
.welcome-emoji {
    font-size: 64px;
    display: block;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
.welcome-title {
    font-family: 'Poppins', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
}
.welcome-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.6;
}

/* ─────────────────────────────────────────────────────────────────────────────
   SIDEBAR BRANDING
───────────────────────────────────────────────────────────────────────────── */
.sidebar-brand {
    font-family: 'Poppins', sans-serif;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, #E11D48, #2563EB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
    display: block;
}
.sidebar-tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 2rem;
    display: block;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid var(--border-glass);
    margin: 1.5rem 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
   INFO CHIP
───────────────────────────────────────────────────────────────────────────── */
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 9999px;
    padding: 4px 12px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #93c5fd;
    margin-bottom: 1rem;
}

/* ─────────────────────────────────────────────────────────────────────────────
   DATAFRAME OVERRIDE
───────────────────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: var(--surface-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: blur(16px);
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Poppins', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load data & build model (cached) ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    df = load_data(DATA_PATH)
    cosine_sim = build_similarity_matrix(df)
    return df, cosine_sim


@st.cache_data(show_spinner=False)
def get_unique_values(df):
    audiences  = sorted(df['target_audience'].dropna().unique().tolist())
    categories = sorted(df['category'].dropna().unique().tolist())
    longevities = sorted(df['longevity'].dropna().unique().tolist())
    return audiences, categories, longevities


# ─── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("🧴 Memuat dataset parfum..."):
    df_full, cosine_sim = get_data()

audiences, categories, longevities = get_unique_values(df_full)

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">✨ AI-Powered · Content-Based Filtering</div>
    <h1 class="hero-title">Fragrance<br>Matchmaker</h1>
    <p class="hero-subtitle">
        Temukan parfum yang paling cocok dengan seleramu.
        Pilih parfum favoritmu, kami akan merekomendasikan yang paling mirip
        menggunakan Cosine Similarity berbasis karakteristik aroma.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-brand">🧴 Fragrance</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-tagline">Matchmaker · Content-Based AI</span>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**🔍 Filter Parfum**")
    st.caption("Filter daftar parfum sebelum memilih rekomendasi")

    # ── Target Audience Filter
    selected_audience = st.selectbox(
        "👥 Target Audience",
        options=["Semua"] + audiences,
        index=0,
        key="sel_audience"
    )

    # ── Dynamic Category (depends on audience)
    if selected_audience == "Semua":
        df_filtered_1 = df_full.copy()
    else:
        df_filtered_1 = df_full[df_full['target_audience'] == selected_audience]

    avail_categories = ["Semua"] + sorted(df_filtered_1['category'].dropna().unique().tolist())

    selected_category = st.selectbox(
        "🌸 Kategori Aroma",
        options=avail_categories,
        index=0,
        key="sel_category"
    )

    # ── Dynamic Longevity (depends on audience + category)
    if selected_category == "Semua":
        df_filtered_2 = df_filtered_1.copy()
    else:
        df_filtered_2 = df_filtered_1[df_filtered_1['category'] == selected_category]

    avail_longevities = ["Semua"] + sorted(df_filtered_2['longevity'].dropna().unique().tolist())

    selected_longevity = st.selectbox(
        "⏱️ Longevity (Ketahanan)",
        options=avail_longevities,
        index=0,
        key="sel_longevity"
    )

    # ── Apply all filters
    df_view = df_filtered_2.copy()
    if selected_longevity != "Semua":
        df_view = df_view[df_view['longevity'] == selected_longevity]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**⚙️ Pengaturan Rekomendasi**")

    top_n = st.slider(
        "🎯 Jumlah Rekomendasi (Top-N)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="slider_topn"
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Stats chips
    total = len(df_full)
    filtered = len(df_view)
    st.markdown(
        f'<div class="info-chip">📊 {filtered:,} dari {total:,} parfum tersedia</div>',
        unsafe_allow_html=True
    )


# ─── MAIN CONTENT ──────────────────────────────────────────────────────────────

# Guard: empty state after filtering
if df_view.empty:
    st.markdown("""
    <div class="welcome-box">
        <span class="welcome-emoji">🔎</span>
        <p class="welcome-title">Tidak Ada Parfum Ditemukan</p>
        <p class="welcome-sub">
            Filter yang kamu pilih tidak menghasilkan parfum apapun.<br>
            Coba longgarkan filter di sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Perfume Selector
perfume_options = sorted(df_view['perfume'].dropna().unique().tolist())

col_main, col_hint = st.columns([3, 1])
with col_main:
    selected_perfume = st.selectbox(
        "🧴 Pilih Parfum Kamu",
        options=perfume_options,
        index=0,
        key="sel_perfume",
        help="Daftar parfum telah disesuaikan dengan filter sidebar"
    )
with col_hint:
    st.markdown("<br>", unsafe_allow_html=True)
    match_row = df_view[df_view['perfume'] == selected_perfume].iloc[0]
    st.markdown(
        f'<div style="padding-top:10px;">'
        f'<span class="rec-tag">{match_row["type"].upper()}</span> '
        f'<span class="rec-tag">{match_row["category"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Selected Perfume Metric Cards
row = df_view[df_view['perfume'] == selected_perfume].iloc[0]

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <span class="metric-icon">🏷️</span>
        <div class="metric-label">Brand</div>
        <p class="metric-value">{row['brand']}</p>
    </div>
    <div class="metric-card">
        <span class="metric-icon">🍶</span>
        <div class="metric-label">Tipe</div>
        <p class="metric-value">{row['type'].upper()}</p>
    </div>
    <div class="metric-card">
        <span class="metric-icon">🌸</span>
        <div class="metric-label">Kategori Aroma</div>
        <p class="metric-value">{row['category']}</p>
    </div>
    <div class="metric-card">
        <span class="metric-icon">👥</span>
        <div class="metric-label">Target & Longevity</div>
        <p class="metric-value">{row['target_audience']} · {row['longevity']}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search Button
search_clicked = st.button("🔍 Temukan Rekomendasi Terbaik", key="btn_search")

# ─── RESULTS ───────────────────────────────────────────────────────────────────
if search_clicked:
    try:
        with st.spinner("⚗️ Menghitung kemiripan aroma..."):
            results = get_recommendations(
                perfume_name=selected_perfume,
                df=df_full,
                cosine_sim=cosine_sim,
                top_n=top_n,
            )

        # Section header
        st.markdown(f"""
        <div class="section-header">
            <h2 class="section-title">✨ Rekomendasi untuk "<em>{selected_perfume}</em>"</h2>
            <span class="section-count">Top {len(results)}</span>
        </div>
        """, unsafe_allow_html=True)

        if results.empty:
            st.warning("Tidak ada rekomendasi yang ditemukan. Coba parfum yang berbeda.")
        else:
            # ── Render card grid via HTML ──────────────────────────────────────
            cards_html = '<div class="rec-grid">'

            for rank, (_, r) in enumerate(results.iterrows(), start=1):
                score = r['similarity_score']
                bar_pct = int(score * 100)

                # Longevity color
                lon_colors = {
                    'light':  ('#FACC15', '#78350F'),
                    'medium': ('#60A5FA', '#1E3A5F'),
                    'strong': ('#F87171', '#450A0A'),
                }
                l_bg, l_text = lon_colors.get(r['longevity'], ('#9CA3AF', '#1F2937'))

                # Audience color
                aud_colors = {
                    'male':   ('#3B82F6', '#1E3A5F'),
                    'female': ('#EC4899', '#4A0017'),
                    'unisex': ('#8B5CF6', '#2E1065'),
                }
                a_bg, a_text = aud_colors.get(r['target_audience'], ('#9CA3AF', '#1F2937'))

                cards_html += (
                    f'<div class="rec-card">'
                    f'<span class="rec-rank">#{rank}</span>'
                    f'<div class="rec-score-bar-wrap">'
                    f'<div class="rec-score-bar" style="width:{bar_pct}%;"></div>'
                    f'</div>'
                    f'<p class="rec-perfume">{r["perfume"]}</p>'
                    f'<p class="rec-brand">{r["brand"]}</p>'
                    f'<div class="rec-tags">'
                    f'<span class="rec-tag">{r["type"].upper()}</span>'
                    f'<span class="rec-tag" style="background:rgba(225,29,72,0.12);border-color:rgba(225,29,72,0.25);color:#fb7185;">{r["category"]}</span>'
                    f'<span class="rec-tag" style="background:{l_bg}22;border-color:{l_bg}44;color:{l_bg};">⏱ {r["longevity"]}</span>'
                    f'<span class="rec-tag" style="background:{a_bg}22;border-color:{a_bg}44;color:{a_bg};">👤 {r["target_audience"]}</span>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span class="rec-score-label">Similarity Score</span>'
                    f'<span class="rec-score-val">{score:.4f}</span>'
                    f'</div>'
                    f'</div>'
                )

            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Full Data Table (expandable) ───────────────────────────────────
            with st.expander("📋 Lihat Tabel Lengkap Rekomendasi", expanded=False):
                display_df = results.copy()
                display_df.index = display_df.index + 1
                display_df.index.name = "Rank"
                display_df.columns = [
                    "Brand", "Perfume", "Type",
                    "Category", "Audience", "Longevity", "Similarity Score"
                ]
                display_df["Similarity Score"] = display_df["Similarity Score"].map("{:.4f}".format)

                st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(60 + len(display_df) * 35, 400),
                )
                st.markdown('</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"❌ {e}. Silakan pilih parfum yang berbeda.")
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {e}")

else:
    # ── Welcome / idle state
    st.markdown("""
    <div class="welcome-box">
        <span class="welcome-emoji">🧴</span>
        <p class="welcome-title">Siap Menemukan Parfummu?</p>
        <p class="welcome-sub">
            Pilih parfum di atas, atur filter di sidebar,<br>
            lalu tekan <strong style="color:#E11D48;">Temukan Rekomendasi Terbaik</strong> untuk memulai.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:4rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.08);">
    <p style="font-family:'DM Sans',sans-serif;font-size:13px;color:#4B5563;margin:0;">
        🧴 <strong style="color:#6B7280;">Fragrance Matchmaker</strong> — 
        Content-Based Filtering · CountVectorizer · Cosine Similarity<br>
        <span style="font-size:11px;">
            Dibuat oleh Tim DSBeginner: Torikh · Alif · Hasan · Study Club 2026
        </span>
    </p>
</div>
""", unsafe_allow_html=True)
