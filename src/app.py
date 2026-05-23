"""
app.py — Fragrance Matchmaker · Liquid Glass & Minimalist Luxury Edition
Design System: Warm Off-White · Bodoni Moda · Jost · Champagne Gold #CA8A04
"""

import os
import sys
import hashlib
import streamlit as st
import pandas as pd

# ─── Path resolution ────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

from recommender import load_data, build_similarity_matrix, get_recommendations

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "perfumes_clean.csv")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fragrance Matchmaker",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Liquid Glass & Minimalist Luxury CSS ────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts: Playfair Display (editorial) + Inter (functional) ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:             #FAFAF9;
    --bg-card:        rgba(255,255,255,0.72);
    --bg-sidebar:     rgba(250,250,249,0.96);
    --glass-blur:     blur(20px);
    --border:         rgba(12,10,9,0.08);
    --border-hover:   rgba(202,138,4,0.35);
    --text-primary:   #0C0A09;
    --text-secondary: #44403C;
    --text-muted:     #78716C;
    --gold:           #CA8A04;
    --gold-light:     rgba(202,138,4,0.12);
    --gold-glow:      0 8px 32px rgba(202,138,4,0.18);
    --shadow-whisper: 0 2px 24px rgba(12,10,9,0.06);
    --shadow-card:    0 4px 32px rgba(12,10,9,0.08);
    --shadow-hover:   0 16px 48px rgba(12,10,9,0.12);
    --radius-sm:      8px;
    --radius-md:      16px;
    --radius-lg:      24px;
    --radius-xl:      32px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome (except collapsed sidebar control) ── */
#MainMenu, footer {
    visibility: hidden;
}
[data-testid="stHeader"] {
    background-color: transparent !important;
    pointer-events: none;
}
[data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    visibility: visible !important;
    pointer-events: auto;
    top: 12px !important;
    left: 16px !important;
}
/* Style the collapsed control (hamburger button) to match the luxury theme */
[data-testid="collapsedControl"] button {
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 200ms ease !important;
    box-shadow: var(--shadow-whisper) !important;
}
[data-testid="collapsedControl"] button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
    background-color: var(--gold-light) !important;
}

/* ── Main container ── */
[data-testid="stAppViewBlockContainer"],
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1320px;
    margin: 0 auto;
}

@media (max-width: 768px) {
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
}

/* ───────────────────────────────────────────────────
   SIDEBAR
─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: var(--glass-blur);
}
[data-testid="stSidebar"] .block-container {
    padding: 2.5rem 1.5rem !important;
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.25rem !important;
    }
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ───────────────────────────────────────────────────
   SELECTBOX & INPUTS — Minimalist underline style
─────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transition: border-color 200ms ease !important;
    padding-left: 0 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stSelectbox"] > div > div:hover {
    border-bottom-color: var(--gold) !important;
    box-shadow: none !important;
}

/* Sidebar selectbox get subtle card styling */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--gold) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--gold) !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
}

/* ── Primary Button — Almost Black, gold on hover ── */
.stButton > button {
    background: var(--text-primary) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    height: 52px !important;
    cursor: pointer;
    transition: all 250ms cubic-bezier(0.25,0.46,0.45,0.94) !important;
    box-shadow: var(--shadow-card) !important;
}
.stButton > button:hover {
    background: var(--gold) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: var(--gold-glow) !important;
    color: #fff !important;
}
.stButton > button:active {
    transform: translateY(0) scale(1) !important;
}

/* ───────────────────────────────────────────────────
   NAV BAR  (top floating glass bar)
─────────────────────────────────────────────────── */
.fm-navbar {
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    background: rgba(250,250,249,0.85);
    backdrop-filter: var(--glass-blur);
}
.fm-navbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.fm-navbar-brand span {
    color: var(--gold);
}
.fm-navbar-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--gold-light);
    border: 1px solid rgba(202,138,4,0.25);
    border-radius: 9999px;
    padding: 5px 14px;
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ───────────────────────────────────────────────────
   HERO SECTION — Cinematic, editorial
─────────────────────────────────────────────────── */
.hero-wrapper {
    position: relative;
    padding: 1rem 0 1.5rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
/* Organic gradient orbs behind the title */
.hero-wrapper::before {
    content: '';
    position: absolute;
    width: 560px; height: 560px;
    background: radial-gradient(circle, rgba(202,138,4,0.10) 0%, transparent 65%);
    top: -120px; right: -80px;
    pointer-events: none;
    animation: orb-float 8s ease-in-out infinite;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(12,10,9,0.05) 0%, transparent 65%);
    bottom: -80px; left: -60px;
    pointer-events: none;
    animation: orb-float 12s ease-in-out infinite reverse;
}
@keyframes orb-float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(12px, -20px) scale(1.05); }
    66%       { transform: translate(-8px, 10px) scale(0.97); }
}
.hero-eyebrow {
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::after {
    content: '';
    flex: 0 0 40px;
    height: 1px;
    background: var(--gold);
    opacity: 0.4;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(48px, 6vw, 80px);
    font-weight: 600;
    line-height: 1.0;
    letter-spacing: -0.025em;
    color: var(--text-primary);
    margin: 0 0 1.5rem 0;
}
.hero-title em {
    font-style: italic;
    color: var(--text-secondary);
}
.hero-subtitle {
    font-family: 'Jost', sans-serif;
    font-size: 18px;
    font-weight: 300;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 560px;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
    margin: 1.5rem 0 0;
}

/* ───────────────────────────────────────────────────
   SELECTED PERFUME — Liquid Glass Profile Card
─────────────────────────────────────────────────── */
.profile-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.5rem 3rem;
    backdrop-filter: var(--glass-blur);
    box-shadow: var(--shadow-card);
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.profile-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(202,138,4,0.4), transparent);
}
.profile-label {
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.profile-name {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0 0 0.25rem 0;
    line-height: 1.2;
}
.profile-brand {
    font-family: 'Jost', sans-serif;
    font-size: 14px;
    font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0 0 2rem 0;
}
.profile-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.profile-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(12,10,9,0.04);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 6px 16px;
    font-family: 'Jost', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
}
.profile-chip.gold {
    background: var(--gold-light);
    border-color: rgba(202,138,4,0.3);
    color: var(--gold);
}

/* ───────────────────────────────────────────────────
   SECTION TITLE
─────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 2rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0;
}
.section-subtitle {
    font-family: 'Jost', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin: 0;
}
.section-count {
    display: inline-flex;
    align-items: center;
    background: var(--gold-light);
    border: 1px solid rgba(202,138,4,0.25);
    border-radius: 9999px;
    padding: 2px 12px;
    font-family: 'Jost', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.06em;
    margin-left: auto;
}

/* ───────────────────────────────────────────────────
   RECOMMENDATION CARDS — Asymmetric gallery grid
─────────────────────────────────────────────────── */
.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 3rem;
    align-items: stretch;
}
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    backdrop-filter: var(--glass-blur);
    box-shadow: var(--shadow-whisper);
    transition: transform 300ms cubic-bezier(0.25,0.46,0.45,0.94),
                box-shadow 300ms ease,
                border-color 300ms ease;
    cursor: pointer;
    position: relative;
}
.rec-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-hover);
    border-color: var(--border-hover);
}
/* Cinematic image area — "bleeds" from top */
.rec-img-wrapper {
    width: 100%;
    height: 200px;
    overflow: hidden;
    position: relative;
}
.rec-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 600ms cubic-bezier(0.25,0.46,0.45,0.94);
    filter: saturate(0.85) brightness(0.97);
}
.rec-card:hover .rec-img {
    transform: scale(1.06);
}
/* Gradient scrim over image */
.rec-img-wrapper::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60%;
    background: linear-gradient(to top, var(--bg-card), transparent);
    pointer-events: none;
}
/* Rank badge floats on image */
.rec-rank {
    position: absolute;
    top: 14px;
    left: 14px;
    z-index: 2;
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-primary);
    background: rgba(250,250,249,0.92);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 4px 12px;
    backdrop-filter: blur(8px);
}
/* Card body */
.rec-body {
    padding: 20px 22px 22px;
}
/* Similarity score bar */
.rec-score-bar-track {
    height: 2px;
    background: var(--border);
    border-radius: 9999px;
    margin-bottom: 16px;
    overflow: hidden;
}
.rec-score-bar-fill {
    height: 2px;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--text-muted), var(--gold));
    transition: width 0.8s cubic-bezier(0.25,0.46,0.45,0.94);
}
.rec-perfume {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    line-height: 1.25;
    letter-spacing: -0.005em;
}
.rec-brand {
    font-family: 'Jost', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 16px 0;
}
.rec-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 14px;
}
.rec-tag {
    display: inline-block;
    white-space: nowrap;
    background: rgba(12,10,9,0.04);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 3px 10px;
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.rec-score-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    margin-top: 4px;
}
.rec-score-label {
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.rec-score-val {
    font-family: 'Jost', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--gold);
}

/* ───────────────────────────────────────────────────
   WELCOME / IDLE STATE
─────────────────────────────────────────────────── */
.welcome-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 6rem 2rem;
    text-align: center;
    backdrop-filter: var(--glass-blur);
    box-shadow: var(--shadow-whisper);
    margin: 2rem 0;
}
.welcome-icon {
    width: 64px; height: 64px;
    margin: 0 auto 1.5rem;
    opacity: 0.18;
    animation: icon-float 4s ease-in-out infinite;
}
@keyframes icon-float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0 0 0.75rem 0;
}
.welcome-sub {
    font-family: 'Jost', sans-serif;
    font-size: 16px;
    font-weight: 300;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 400px;
    margin: 0 auto;
}

/* ───────────────────────────────────────────────────
   SIDEBAR BRANDING
─────────────────────────────────────────────────── */
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    display: block;
    margin-bottom: 2px;
}
.sidebar-tagline {
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 2rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
.sidebar-section-label {
    font-family: 'Jost', sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: block;
}
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(12,10,9,0.04);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 5px 14px;
    font-family: 'Jost', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
}

/* ───────────────────────────────────────────────────
   TABLE WRAPPER
─────────────────────────────────────────────────── */
.table-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    backdrop-filter: var(--glass-blur);
    box-shadow: var(--shadow-whisper);
    margin-bottom: 2rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.5rem !important;
    backdrop-filter: var(--glass-blur);
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 24px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 13px !important;
    letter-spacing: 0.06em !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-card) !important;
    backdrop-filter: var(--glass-blur);
}

/* ── Alert/warning ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}

/* ── Page load entrance animation ── */
@keyframes fade-up {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-wrapper, .profile-card, .rec-grid, .welcome-box {
    animation: fade-up 0.5s cubic-bezier(0.25,0.46,0.45,0.94) both;
}
.rec-card:nth-child(1) { animation: fade-up 0.4s 0.05s ease both; }
.rec-card:nth-child(2) { animation: fade-up 0.4s 0.10s ease both; }
.rec-card:nth-child(3) { animation: fade-up 0.4s 0.15s ease both; }
.rec-card:nth-child(4) { animation: fade-up 0.4s 0.20s ease both; }
.rec-card:nth-child(5) { animation: fade-up 0.4s 0.25s ease both; }

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── Load data & model (cached) ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    df = load_data(DATA_PATH)
    cosine_sim = build_similarity_matrix(df)
    return df, cosine_sim

@st.cache_data(show_spinner=False)
def get_unique_values(df):
    audiences   = sorted(df['target_audience'].dropna().unique().tolist())
    categories  = sorted(df['category'].dropna().unique().tolist())
    longevities = sorted(df['longevity'].dropna().unique().tolist())
    return audiences, categories, longevities


# ─── Load ────────────────────────────────────────────────────────────────────
with st.spinner("Memuat koleksi parfum..."):
    df_full, cosine_sim = get_data()

audiences, categories, longevities = get_unique_values(df_full)


# ─── NAVBAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fm-navbar">
    <span class="fm-navbar-brand">Fragrance<span>.</span></span>
    <span class="fm-navbar-pill">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>
        </svg>
        AI-Powered Matching
    </span>
</div>
""", unsafe_allow_html=True)


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">Content-Based Filtering</div>
    <h1 class="hero-title">
        Temukan Aroma<br>
        <em>Sempurnamu</em>
    </h1>
    <p class="hero-subtitle">
        Pilih parfum favoritmu — algoritma kami akan menemukan
        yang paling mirip dari 900+ koleksi, berdasarkan
        karakteristik aroma, longevity, dan profil.
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-brand">Fragrance.</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-tagline">Matchmaker · Study Club 2026</span>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section-label">Filter Koleksi</span>', unsafe_allow_html=True)

    selected_audience = st.selectbox(
        "Target Audience",
        options=["Semua"] + audiences,
        index=0,
        key="sel_audience"
    )

    if selected_audience == "Semua":
        df_filtered_1 = df_full.copy()
    else:
        df_filtered_1 = df_full[df_full['target_audience'] == selected_audience]

    avail_categories = ["Semua"] + sorted(df_filtered_1['category'].dropna().unique().tolist())
    selected_category = st.selectbox(
        "Kategori Aroma",
        options=avail_categories,
        index=0,
        key="sel_category"
    )

    if selected_category == "Semua":
        df_filtered_2 = df_filtered_1.copy()
    else:
        df_filtered_2 = df_filtered_1[df_filtered_1['category'] == selected_category]

    avail_longevities = ["Semua"] + sorted(df_filtered_2['longevity'].dropna().unique().tolist())
    selected_longevity = st.selectbox(
        "Longevity",
        options=avail_longevities,
        index=0,
        key="sel_longevity"
    )

    df_view = df_filtered_2.copy()
    if selected_longevity != "Semua":
        df_view = df_view[df_view['longevity'] == selected_longevity]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-section-label">Pengaturan Rekomendasi</span>', unsafe_allow_html=True)

    top_n = st.slider(
        "Jumlah Rekomendasi (Top-N)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="slider_topn"
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    total    = len(df_full)
    filtered = len(df_view)
    st.markdown(
        f'<div class="info-chip">'
        f'<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="opacity:0.5">'
        f'<path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>'
        f'</svg>'
        f'{filtered:,} dari {total:,} parfum</div>',
        unsafe_allow_html=True
    )


# ─── GUARD: Empty filter state ───────────────────────────────────────────────
if df_view.empty:
    st.markdown("""
    <div class="welcome-box">
        <svg class="welcome-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M11 6.5A.5.5 0 0 1 11.5 6h1a.5.5 0 0 1 0 1h-1a.5.5 0 0 1-.5-.5zm0 3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 0 1h-1a.5.5 0 0 1-.5-.5zm-3 3.5a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1H8zm0 2.5a.5.5 0 0 1 0-1h4a.5.5 0 0 1 0 1H8z"/>
            <path d="M3 0h10a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm0 1a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H3z"/>
        </svg>
        <p class="welcome-title">Tidak Ada Parfum</p>
        <p class="welcome-sub">Filter yang dipilih tidak menghasilkan parfum apapun. Coba longgarkan filter di sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── PERFUME SELECTOR ────────────────────────────────────────────────────────
perfume_options = sorted(df_view['perfume'].dropna().unique().tolist())

col_sel, col_tag = st.columns([4, 1])
with col_sel:
    selected_perfume = st.selectbox(
        "Pilih Parfum",
        options=perfume_options,
        index=0,
        key="sel_perfume",
        help="Daftar parfum disesuaikan dengan filter aktif"
    )
with col_tag:
    match_row = df_view[df_view['perfume'] == selected_perfume].iloc[0]
    st.markdown(
        f'<div style="padding-top:1.8rem; display:flex; flex-wrap:wrap; gap:6px;">'
        f'<span class="rec-tag">{match_row["type"].upper()}</span>'
        f'<span class="rec-tag" style="color:var(--gold);border-color:rgba(202,138,4,0.3);background:var(--gold-light);">{match_row["category"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ─── SELECTED PERFUME PROFILE CARD ───────────────────────────────────────────
row = df_view[df_view['perfume'] == selected_perfume].iloc[0]

# Longevity display chip color
lon_style = {
    'light':  'background:rgba(234,179,8,0.1);border-color:rgba(234,179,8,0.3);color:#92400e;',
    'medium': 'background:rgba(14,165,233,0.1);border-color:rgba(14,165,233,0.3);color:#0369a1;',
    'strong': 'background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.25);color:#991b1b;',
}
lon_chip = lon_style.get(row['longevity'], '')

st.markdown(f"""
<div class="profile-card">
    <div class="profile-label">Parfum Terpilih</div>
    <p class="profile-name">{row['perfume'].title()}</p>
    <p class="profile-brand">{row['brand']}</p>
    <div class="profile-meta">
        <span class="profile-chip gold">{row['type'].upper()}</span>
        <span class="profile-chip">{row['category']}</span>
        <span class="profile-chip" style="{lon_chip}">{row['longevity'].title()} Longevity</span>
        <span class="profile-chip">{row['target_audience'].title()}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── SEARCH BUTTON ───────────────────────────────────────────────────────────
search_clicked = st.button("Temukan Rekomendasi", key="btn_search")


# ─── RESULTS ─────────────────────────────────────────────────────────────────
if search_clicked:
    try:
        with st.spinner("Menghitung kemiripan aroma..."):
            results = get_recommendations(
                perfume_name=selected_perfume,
                df=df_full,
                cosine_sim=cosine_sim,
                top_n=top_n,
            )

        # Section header
        st.markdown(f"""
        <div class="section-header">
            <h2 class="section-title">Rekomendasi</h2>
            <span class="section-subtitle">untuk &ldquo;{selected_perfume.title()}&rdquo;</span>
            <span class="section-count">Top {len(results)}</span>
        </div>
        """, unsafe_allow_html=True)

        if results.empty:
            st.warning("Tidak ada rekomendasi ditemukan. Coba parfum yang berbeda.")
        else:
            # ── Build card grid HTML ────────────────────────────────────────
            cards_html = '<div class="rec-grid">'

            for rank, (_, r) in enumerate(results.iterrows(), start=1):
                score    = r['similarity_score']
                bar_pct  = int(score * 100)

                # Category tag with subtle tint
                cat_lower = r['category'].lower()
                if 'floral' in cat_lower:
                    tag_style = 'background:rgba(236,72,153,0.08);border-color:rgba(236,72,153,0.2);color:#9d174d;'
                elif 'woody' in cat_lower or 'wood' in cat_lower:
                    tag_style = 'background:rgba(120,80,40,0.08);border-color:rgba(120,80,40,0.2);color:#713f12;'
                elif 'oud' in cat_lower:
                    tag_style = 'background:rgba(161,98,7,0.08);border-color:rgba(161,98,7,0.2);color:#78350f;'
                elif 'fresh' in cat_lower or 'citrus' in cat_lower:
                    tag_style = 'background:rgba(14,165,233,0.08);border-color:rgba(14,165,233,0.2);color:#0c4a6e;'
                elif 'oriental' in cat_lower or 'amber' in cat_lower:
                    tag_style = 'background:rgba(202,138,4,0.08);border-color:rgba(202,138,4,0.2);color:#713f12;'
                else:
                    tag_style = ''

                # Longevity badge
                lon_map = {
                    'light':  ('rgba(234,179,8,0.1)', 'rgba(234,179,8,0.25)', '#92400e'),
                    'medium': ('rgba(14,165,233,0.1)', 'rgba(14,165,233,0.25)', '#0369a1'),
                    'strong': ('rgba(239,68,68,0.08)', 'rgba(239,68,68,0.2)', '#991b1b'),
                }
                lon_bg, lon_border, lon_color = lon_map.get(
                    r['longevity'], ('rgba(12,10,9,0.04)', 'var(--border)', 'var(--text-secondary)')
                )

                cards_html += (
                    f'<div class="rec-card" style="padding-top: 1rem;">'
                    f'  <div class="rec-body">'
                    f'    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1.5rem;">'
                    f'      <span class="rec-rank" style="position:static; padding:4px 10px; font-size:12px; font-weight:600; background:var(--gold-light); color:var(--gold); border:1px solid rgba(202,138,4,0.3); border-radius:999px;">#{rank:02d}</span>'
                    f'      <div class="rec-score-bar-track" style="width:120px; margin-bottom:0;">'
                    f'        <div class="rec-score-bar-fill" style="width:{bar_pct}%;"></div>'
                    f'      </div>'
                    f'    </div>'
                    f'    <p class="rec-perfume">{r["perfume"].title()}</p>'
                    f'    <p class="rec-brand">{r["brand"]}</p>'
                    f'    <div class="rec-tags">'
                    f'      <span class="rec-tag">{r["type"].upper()}</span>'
                    f'      <span class="rec-tag" style="{tag_style}">{r["category"]}</span>'
                    f'      <span class="rec-tag" style="background:{lon_bg};border-color:{lon_border};color:{lon_color};">'
                    f'        {r["longevity"].title()}'
                    f'      </span>'
                    f'      <span class="rec-tag">{r["target_audience"].title()}</span>'
                    f'    </div>'
                    f'    <div class="rec-score-footer">'
                    f'      <span class="rec-score-label">Similarity</span>'
                    f'      <span class="rec-score-val">{score:.1%}</span>'
                    f'    </div>'
                    f'  </div>'
                    f'</div>'
                )

            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Full data table ─────────────────────────────────────────────
            with st.expander("Lihat Tabel Lengkap Rekomendasi", expanded=False):
                display_df = results.copy()
                display_df.index = display_df.index + 1
                display_df.index.name = "Rank"
                display_df.columns = [
                    "Brand", "Perfume", "Type",
                    "Category", "Audience", "Longevity", "Similarity Score"
                ]
                display_df["Similarity Score"] = display_df["Similarity Score"].map("{:.2%}".format)
                st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(60 + len(display_df) * 35, 400),
                )
                st.markdown('</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Parfum tidak ditemukan dalam dataset: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    # ── IDLE / WELCOME STATE ────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-box">
        <svg class="welcome-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 4V2h10v2h3a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3zm0 2H5v14h14V6h-2v2H7V6zm2-2v2h6V4H9z"/>
        </svg>
        <p class="welcome-title">Siap Menemukan Aromamu?</p>
        <p class="welcome-sub">
            Pilih parfum di atas, atur filter di sidebar,
            lalu tekan <strong>Temukan Rekomendasi</strong> untuk memulai.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:6rem;padding-top:2rem;border-top:1px solid rgba(12,10,9,0.06);">
    <p style="font-family:'Jost',sans-serif;font-size:12px;color:#A8A29E;letter-spacing:0.06em;margin:0;">
        FRAGRANCE MATCHMAKER &nbsp;&middot;&nbsp; Content-Based Filtering &nbsp;&middot;&nbsp; Cosine Similarity
    </p>
    <p style="font-family:'Jost',sans-serif;font-size:11px;color:#C7C3BF;letter-spacing:0.04em;margin:6px 0 0;">
        Torikh &middot; Alif &middot; Hasan &nbsp;&mdash;&nbsp; Study Club DS Beginner 2026
    </p>
</div>
""", unsafe_allow_html=True)
