# Milestone 01: Data Acquisition & Preprocessing

## Tujuan Milestone

Milestone 1 bertujuan untuk menyiapkan data pipeline yang dapat dijalankan ulang untuk dua dataset klasifikasi jamur:

1. Mengambil dan menyimpan UCI Mushroom Dataset resmi ke struktur proyek.
2. Membaca Secondary Mushroom Dataset dari lokasi standar proyek.
3. Melakukan inspeksi awal terhadap data mentah.
4. Mengimplementasikan preprocessing yang konsisten, bebas data leakage pada tahap fitting preprocessor, dan menghasilkan artefak siap pakai untuk modeling.
5. Memastikan pipeline preprocessing tervalidasi melalui automated tests.

## Sumber dan Lokasi Dataset

### UCI Mushroom Dataset

- Sumber: UCI Machine Learning Repository melalui package `ucimlrepo`
- Dataset ID: `73`
- Script unduh: [src/download_data.py](D:\my-kisah\yayayaya\college\machine-learning\src\download_data.py)
- Lokasi raw file: `data/raw/uci/mushroom.csv`

### Secondary Mushroom Dataset

- Sumber: file manual yang disediakan terpisah
- Lokasi raw file: `data/raw/secondary/secondary_mushroom.csv`
- Format file saat implementasi: delimiter `;`

## Hasil Inspeksi Awal

Inspeksi awal dilakukan menggunakan [src/inspect_data.py](D:\my-kisah\yayayaya\college\machine-learning\src\inspect_data.py).

### UCI Mushroom Dataset

- Shape: `8124 x 23`
- Semua kolom bertipe kategorikal (`object`)
- Missing values:
  - `stalk-root`: `2480`
- Duplicate rows: `0`
- Distribusi target mentah:
  - `e`: `4208`
  - `p`: `3916`

### Secondary Mushroom Dataset

- Shape: `61069 x 21`
- Tipe data:
  - `3` kolom numerik
  - `18` kolom kategorikal
- Missing values utama:
  - `veil-type`: `57892`
  - `spore-print-color`: `54715`
  - `veil-color`: `53656`
  - `stem-root`: `51538`
  - `stem-surface`: `38124`
  - `gill-spacing`: `25063`
  - `cap-surface`: `14120`
  - `gill-attachment`: `9884`
  - `ring-type`: `2471`
- Exact duplicate rows: `146`
- Distribusi target mentah:
  - `e`: `27181`
  - `p`: `33888`

## Keputusan Penanganan Missing Value dan Duplikasi

### UCI Mushroom Dataset

- Kolom `stalk-root` dipertahankan.
- Missing value kategorikal diisi dengan kategori `unknown`.
- Tidak ada baris yang dihapus.

### Secondary Mushroom Dataset

- Exact duplicate rows dihapus sebelum split.
- Kolom dengan missing value lebih dari `80%` ditentukan dari data train saja lalu di-drop dari train dan test untuk menjaga aturan anti-leakage pada keputusan berbasis data.
- Missing value kategorikal lain diisi dengan `unknown`.
- Missing value numerik diisi dengan median.

## Langkah Preprocessing yang Diimplementasikan

Implementasi ada di [src/preprocessing.py](D:\my-kisah\yayayaya\college\machine-learning\src\preprocessing.py) dan dijalankan melalui [src/run_preprocessing.py](D:\my-kisah\yayayaya\college\machine-learning\src\run_preprocessing.py).

### UCI Mushroom Dataset

1. Membaca `data/raw/uci/mushroom.csv`
2. Memisahkan fitur dan target `class`
3. Encode target dengan mapping `e=0`, `p=1`
4. Train/test split `80:20` dengan `stratify=y` dan `random_state=42`
5. Kategorikal:
   - `SimpleImputer(strategy="constant", fill_value="unknown")`
   - `OneHotEncoder(handle_unknown="ignore")`
6. Menyimpan artefak preprocessing ke `data/processed/uci/`

### Secondary Mushroom Dataset

1. Membaca `data/raw/secondary/secondary_mushroom.csv` dengan auto-detect separator
2. Menghapus `146` exact duplicate rows
3. Memisahkan fitur dan target `class`
4. Encode target dengan mapping `e=0`, `p=1`
5. Train/test split `80:20` dengan `stratify=y` dan `random_state=42`
6. Menentukan kolom dengan missing rate `>80%` dari data train lalu drop dari train dan test
7. Numerik:
   - `SimpleImputer(strategy="median")`
8. Kategorikal:
   - `SimpleImputer(strategy="constant", fill_value="unknown")`
   - `OneHotEncoder(handle_unknown="ignore")`
9. Menyimpan artefak preprocessing ke `data/processed/secondary/`

## Metode Encoding Target dan Fitur

### Target Encoding

- `e -> 0`
- `p -> 1`

### Feature Encoding

- Fitur numerik: `SimpleImputer(strategy="median")`
- Fitur kategorikal: `SimpleImputer(strategy="constant", fill_value="unknown")` lalu `OneHotEncoder(handle_unknown="ignore")`
- Komposisi preprocessing menggunakan:
  - `ColumnTransformer`
  - `Pipeline`

Catatan:

- `LabelEncoder` tidak digunakan untuk fitur input.
- Split dilakukan sebelum preprocessor di-fit.
- Preprocessor hanya di-fit pada `X_train`.

## Konfigurasi Train/Test Split

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

## Hasil Preprocessing

### UCI Mushroom Dataset

- Shape sebelum cleaning: `8124 x 23`
- Shape sesudah cleaning: `8124 x 23`
- Duplicate yang dihapus: `0`
- Kolom yang di-drop: tidak ada
- Ukuran split:
  - train: `6499`
  - test: `1625`
- Jumlah fitur sebelum encoding: `22`
- Jumlah fitur sesudah encoding: `117`
- Missing value sesudah preprocessing:
  - `X_train_total = 0`
  - `X_test_total = 0`
- Distribusi kelas ter-encode:
  - full: `0=4208`, `1=3916`
  - train: `0=3366`, `1=3133`
  - test: `0=842`, `1=783`

### Secondary Mushroom Dataset

- Shape sebelum cleaning: `61069 x 21`
- Shape sesudah cleaning: `60923 x 17`
- Duplicate yang dihapus: `146`
- Kolom yang di-drop:
  - `spore-print-color`
  - `stem-root`
  - `veil-color`
  - `veil-type`
- Ukuran split:
  - train: `48738`
  - test: `12185`
- Jumlah fitur sebelum encoding: `16`
- Jumlah fitur sesudah encoding: `105`
- Missing value sesudah preprocessing:
  - `X_train_total = 0`
  - `X_test_total = 0`
- Distribusi kelas ter-encode setelah deduplikasi:
  - full: `0=27181`, `1=33742`
  - train: `0=21745`, `1=26993`
  - test: `0=5436`, `1=6749`

## Daftar File Output yang Dihasilkan

Setiap dataset menghasilkan artefak berikut:

- `X_train.npz`
- `X_test.npz`
- `y_train.npy`
- `y_test.npy`
- `preprocessor.joblib`
- `feature_names.csv`
- `preprocessing_summary.json`

Lokasi output:

- `data/processed/uci/`
- `data/processed/secondary/`

Struktur output saat ini:

```text
data/processed/
|-- secondary/
|   |-- feature_names.csv
|   |-- preprocessing_summary.json
|   |-- preprocessor.joblib
|   |-- X_test.npz
|   |-- X_train.npz
|   |-- y_test.npy
|   `-- y_train.npy
`-- uci/
    |-- feature_names.csv
    |-- preprocessing_summary.json
    |-- preprocessor.joblib
    |-- X_test.npz
    |-- X_train.npz
    |-- y_test.npy
    `-- y_train.npy
```

## Hasil Pengujian Preprocessing

Automated tests ada di [tests/test_preprocessing.py](D:\my-kisah\yayayaya\college\machine-learning\tests\test_preprocessing.py).

Yang diuji:

- seluruh file output tersedia untuk kedua dataset
- target ter-encode ke `0` dan `1`
- bentuk `X_train`, `X_test`, `y_train`, `y_test` konsisten
- jumlah feature names sesuai jumlah kolom hasil transform
- tidak ada missing value setelah preprocessing
- aturan khusus UCI sesuai spesifikasi
- aturan khusus secondary sesuai spesifikasi

Hasil test terakhir:

```text
6 passed in 2.98s
```

## Cara Menjalankan Ulang

### 1. Install dependency

```powershell
python -m pip install -r requirements.txt
```

### 2. Download ulang UCI dataset resmi

```powershell
python src\download_data.py
```

### 3. Pastikan secondary dataset tersedia

File harus tersedia di:

```text
data/raw/secondary/secondary_mushroom.csv
```

### 4. Jalankan preprocessing kedua dataset

```powershell
python src\run_preprocessing.py --dataset all
```

Untuk satu dataset saja:

```powershell
python src\run_preprocessing.py --dataset uci
python src\run_preprocessing.py --dataset secondary
```

### 5. Jalankan test preprocessing

```powershell
python -m pytest -q
```

## Status Akhir Milestone

Status milestone saat ini: `COMPLETED`

Alasan status `COMPLETED` valid:

- preprocessing UCI berhasil
- preprocessing secondary berhasil
- seluruh artefak output tersedia untuk kedua dataset
- seluruh test preprocessing lulus

## Handoff

### Handoff untuk Person 2

- Gunakan artefak di `data/processed/uci/` untuk baseline modeling pada UCI dataset.
- Muat `preprocessor.joblib` dan `feature_names.csv` jika perlu interpretasi fitur hasil one-hot encoding.
- Fokus berikutnya: benchmarking model klasifikasi dan evaluasi performa.

### Handoff untuk Person 3

- Gunakan artefak di `data/processed/secondary/` untuk baseline modeling pada secondary dataset.
- Perhatikan bahwa kolom berikut sudah di-drop karena missing `>80%`:
  - `spore-print-color`
  - `stem-root`
  - `veil-color`
  - `veil-type`
- Fokus berikutnya: eksperimen model, analisis pentingnya fitur, dan perbandingan dengan dataset UCI.

### Handoff untuk Person 4

- Validasi integrasi pipeline dengan tahap modeling dan evaluasi.
- Gunakan `preprocessing_summary.json` dari masing-masing dataset sebagai sumber metadata eksperimen.
- Fokus berikutnya:
  - konsolidasi hasil modeling lintas dataset
  - pelaporan eksperimen
  - quality check artefak sebelum milestone berikutnya
