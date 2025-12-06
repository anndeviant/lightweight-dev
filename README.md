# Lightweight-Dev

Project pelatihan model LightGBM dengan teknik sampling dan hyperparameter optimization menggunakan Optuna with Python 3.11.9.

## Struktur Folder

```
lightweight-dev/
├── dataset_sampling/         # Folder untuk proses sampling dataset
│   ├── pipreqs.ipynb # Install depedencies
│   ├── requirements.txt      # Dependencies yang diperlukan
│   └── sampling_3_rus-smote.ipynb
├── dataset_splitted/         # Folder berisi dataset yang sudah di-split dan scaled
│   ├── train_split_scaled.csv
│   ├── val_split_scaled.csv
│   ├── test_split_scaled.csv
│   ├── scaler_info.json
│   └── read_pkl.py
└── lightgbm/                 # Folder untuk training model LightGBM
    ├── lightgbm_1_baseline_.ipynb
    ├── lightgbm_1_baseline_optuna.ipynb
    ├── lightgbm_1_smote-rus_.ipynb
    ├── lightgbm_1_smote-rus_optuna.ipynb
    ├── optuna_2_optuna_.ipynb
    ├── optuna_2_optuna_smote-rus.ipynb
    ├── saved_hyperparams_baseline/
    ├── saved_hyperparams_smote-rus/
    ├── saved_model_baseline/
    ├── saved_model_baseline_optuna/
    ├── saved_model_smote-rus/
    └── saved_model_smote-rus_optuna/
```

## Ringkasan Workflow

```
1. Sampling (sampling_3_rus-smote.ipynb)
   ├─> Install requirements (pipreqs.ipynb)
   └─> Apply RUS-SMOTE sampling (Run All)

2. Training Baseline
   ├─> Baseline tanpa sampling (lightgbm_1_baseline_.ipynb)
   └─> Baseline dengan SMOTE-RUS (lightgbm_1_smote-rus_.ipynb)

3. Hyperparameter Optimization
   ├─> Optuna Baseline (optuna_2_optuna_.ipynb)
   │   └─> Masukkan params ke lightgbm_1_baseline_optuna.ipynb
   │
   └─> Optuna SMOTE-RUS (optuna_2_optuna_smote-rus.ipynb)
       └─> Masukkan params ke lightgbm_1_smote-rus_optuna.ipynb
```

## Alur Eksekusi

### Step 1: Install Dependencies & Data Sampling

**File:** `dataset_sampling/pipreqs.ipynb`

1. **Install Requirements**
   - Python 3.11.9
   - Jalankan cell pertama untuk install dependencies:
   ```python
   %pip install -r requirements.txt
   ```
2. **Jalankan RUS-SMOTE Sampling**
   - Run All

---

### Step 2: Training Model Baseline

#### 2.1. Training Baseline Tanpa Sampling

**File:** `lightgbm/lightgbm_1_baseline_.ipynb`

1. Jalankan semua cell secara berurutan
2. Model akan di-training dengan data yang sudah di-scaled (tanpa sampling)
3. Model dan hasil evaluasi akan disimpan di folder `saved_model_baseline/`

#### 2.2. Training Baseline dengan SMOTE-RUS

**File:** `lightgbm/lightgbm_1_smote-rus_.ipynb`

1. Jalankan semua cell secara berurutan
2. Model akan di-training dengan data yang sudah di-sampling menggunakan RUS-SMOTE
3. Model dan hasil evaluasi akan disimpan di folder `saved_model_smote-rus/`

---

### Step 3: Hyperparameter Optimization dengan Optuna

#### 3.1. Optuna untuk Baseline (Tanpa Sampling)

**File:** `lightgbm/optuna_2_optuna_.ipynb`

1. Jalankan semua cell secara berurutan
2. Optuna akan mencari hyperparameter terbaik untuk model baseline
3. Best hyperparameters akan disimpan di folder `saved_hyperparams_baseline/`

**Setelah mendapatkan hyperparameters:**

- Copy hyperparameters terbaik dari hasil Optuna
- Buka file `lightgbm/lightgbm_1_baseline_optuna.ipynb`
- **Masukkan hyperparameters** yang didapat dari Optuna ke dalam cell training
- Jalankan semua cell untuk training model dengan hyperparameters optimal
- Model optimized akan disimpan di folder `saved_model_baseline_optuna/`

#### 3.2. Optuna untuk SMOTE-RUS

**File:** `lightgbm/optuna_2_optuna_smote-rus.ipynb`

1. Jalankan semua cell secara berurutan
2. Optuna akan mencari hyperparameter terbaik untuk model dengan SMOTE-RUS
3. Best hyperparameters akan disimpan di folder `saved_hyperparams_smote-rus/`

**Setelah mendapatkan hyperparameters:**

- Copy hyperparameters terbaik dari hasil Optuna
- Buka file `lightgbm/lightgbm_1_smote-rus_optuna.ipynb`
- **Masukkan hyperparameters** yang didapat dari Optuna ke dalam cell training
- Jalankan semua cell untuk training model dengan hyperparameters optimal
- Model optimized akan disimpan di folder `saved_model_smote-rus_optuna/`

---

## Dependencies

Semua dependencies yang diperlukan ada di `dataset_sampling/requirements.txt`:

- pandas
- numpy
- imbalanced-learn
- scikit-learn
- lightgbm
- optuna
- optuna-integration[lightgbm]
- matplotlib

**Lightweight Development Environment** December 2025
