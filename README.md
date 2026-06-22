# Milestone 1: Data Acquisition & Preprocessing

Proyek ini menyiapkan struktur dasar untuk klasifikasi jamur dengan dua dataset:

1. UCI Mushroom Dataset
2. Secondary Mushroom Dataset

Fokus tahap ini masih pada akuisisi data dan inspeksi awal. Belum ada encoding, feature engineering, atau modeling.

## Struktur Proyek

```text
machine-learning/
|-- data/
|   |-- raw/
|   |   |-- uci/
|   |   `-- secondary/
|   `-- processed/
|       |-- uci/
|       `-- secondary/
|-- notebooks/
|-- reports/
|   `-- tables/
|-- src/
|   |-- __init__.py
|   `-- inspect_data.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Penempatan Dataset

Letakkan file CSV pada lokasi berikut:

- `data/raw/uci/mushroom.csv`
- `data/raw/secondary/secondary_mushroom.csv`

Script inspeksi juga mencoba beberapa nama file alternatif yang umum jika nama file berbeda.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan Inspeksi Data

Default:

```powershell
python src\inspect_data.py
```

Jika nama file atau lokasi berbeda:

```powershell
python src\inspect_data.py --uci-path data\raw\uci\nama_file_uci.csv --secondary-path data\raw\secondary\nama_file_secondary.csv
```

## Output Script

Untuk masing-masing dataset, script akan menampilkan:

- shape
- lima baris pertama
- info tipe data
- missing values per kolom
- jumlah duplicate rows
- distribusi target class
