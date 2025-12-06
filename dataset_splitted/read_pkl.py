import pickle
import numpy as np
import pandas as pd  # Opsional, biar tampilan lebih rapi
from pathlib import Path

# portabel
script_dir = Path(__file__).parent
nama_file_pkl = script_dir / "scaler_params.pkl"

try:
    with open(nama_file_pkl, "rb") as file:
        scaler = pickle.load(file)

        print("=== INFORMASI SCALER ===")
        print(f"Tipe Scaler: {type(scaler).__name__}")

        # 1. Melihat Rata-rata (Mean) yang dipelajari dari data Training
        # Ini adalah nilai 'u' dalam rumus z = (x - u) / s
        print("\n1. Mean (Rata-rata per fitur):")
        print(scaler.mean_)

        # 2. Melihat Skala (Standar Deviasi)
        # Ini adalah nilai 's' dalam rumus z = (x - u) / s
        print("\n2. Scale (Standar Deviasi per fitur):")
        print(scaler.scale_)

        # 3. Melihat Variansi (Variance)
        print("\n3. Variance (Variansi per fitur):")
        print(scaler.var_)

        # 4. Informasi Tambahan
        print("\n4. Info Lain:")
        if hasattr(scaler, "n_features_in_"):
            print(f"   - Jumlah fitur yang diproses: {scaler.n_features_in_}")
        if hasattr(scaler, "n_samples_seen_"):
            print(
                f"   - Jumlah sampel data training yang dilihat: {scaler.n_samples_seen_}"
            )

        # --- OPSIONAL: Tampilan lebih cantik menggunakan Pandas ---
        # Jika Anda ingin melihat angka ini dipasangkan dengan nama fiturnya
        # (Asumsi fitur Anda berurutan sesuai dataset asli)
        print("\n=== TABEL RINGKASAN (Preview) ===")
        summary = pd.DataFrame(
            {
                "Mean": scaler.mean_,
                "Scale (StdDev)": scaler.scale_,
                "Variance": scaler.var_,
            }
        )
        print(summary.head())  # Menampilkan 5 fitur pertama saja agar tidak kepanjangan

except FileNotFoundError:
    print(f"Error: File '{nama_file_pkl}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi kesalahan saat membaca file: {e}")
