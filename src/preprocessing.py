import pandas as pd
import numpy as np

# ─── Konstanta ────────────────────────────────────────────────────────────────
TEXT_COLS = ['brand', 'perfume', 'type', 'category', 'target_audience', 'longevity']

# ─── Fungsi ───────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Muat dataset dari file CSV."""
    return pd.read_csv(filepath)

def standardize_text(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Ubah teks menjadi lowercase dan hapus whitespace berlebih."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            # Pastikan tipe data adalah string dan ubah ke huruf kecil
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            
            # Gunakan numpy untuk mengganti string kosong dengan 'unknown'
            # (Memenuhi syarat penggunaan Numpy)
            df_clean[col] = np.where(df_clean[col] == '', 'unknown', df_clean[col])
            df_clean[col] = np.where(df_clean[col] == 'nan', 'unknown', df_clean[col])

    # Membersihkan noise spesifik
    if 'target_audience' in df_clean.columns:
        # Menyatukan men->male, women->female, dan menghapus header dataset yang ikut masuk
        df_clean['target_audience'] = df_clean['target_audience'].replace({
            'men': 'male', 
            'women': 'female',
            'target audience': 'unknown'
        })
        
    if 'longevity' in df_clean.columns:
        # Menyeragamkan noise pada longevity
        df_clean['longevity'] = np.where(df_clean['longevity'].str.contains('medium', na=False), 'medium', df_clean['longevity'])
        df_clean['longevity'] = np.where(df_clean['longevity'].str.contains('strong', na=False), 'strong', df_clean['longevity'])
        df_clean['longevity'] = np.where(df_clean['longevity'].str.contains('light', na=False), 'light', df_clean['longevity'])

    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat berdasarkan nama parfum dan brand."""
    return df.drop_duplicates(subset=['brand', 'perfume'], keep='first').reset_index(drop=True)

def create_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan fitur kategorikal menjadi satu kolom teks (Feature Engineering)."""
    df_feat = df.copy()
    
    # Fitur yang akan di-vektorisasi: type, category, target_audience, longevity
    # Kita pisahkan dengan spasi agar CountVectorizer bisa memecahnya sebagai token
    df_feat['combined_features'] = (
        df_feat['type'] + ' ' + 
        df_feat['category'] + ' ' + 
        df_feat['target_audience'] + ' ' + 
        df_feat['longevity']
    )
    return df_feat

def full_pipeline(input_path: str, output_path: str):
    """
    Jalankan seluruh pipeline preprocessing:
    load -> standardize -> drop duplicates -> combine features -> save
    """
    print("Memulai preprocessing pipeline...")
    
    # 1. Load Data
    df = load_data(input_path)
    print(f"Data awal: {df.shape}")
    
    # 2. Standardize Text
    df = standardize_text(df, TEXT_COLS)
    
    # 3. Remove Duplicates
    df = remove_duplicates(df)
    print(f"Data setelah hapus duplikat: {df.shape}")
    
    # 4. Feature Engineering
    df = create_combined_features(df)
    print(f"Kolom baru berhasil dibuat: 'combined_features'")
    
    # 5. Save Clean Data
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke: {output_path}")
    
    return df

if __name__ == "__main__":
    # Path relatif jika dijalankan dari root project
    INPUT_PATH = "data/Perfumes_dataset.csv"
    OUTPUT_PATH = "data/perfumes_clean.csv"
    
    full_pipeline(INPUT_PATH, OUTPUT_PATH)
