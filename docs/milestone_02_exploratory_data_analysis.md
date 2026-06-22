# Milestone 02: Exploratory Data Analysis

## Status

Status milestone saat ini: `COMPLETED`

Status ini valid karena:

- seluruh grafik dibuat dari dua dataset aktual di repository
- seluruh tabel dibuat dari dua dataset aktual di repository
- EDA mengikuti aturan cleaning Milestone 1
- analisis yang memengaruhi feature selection menggunakan train split `80:20`, `stratify=y`, `random_state=42`
- seluruh test EDA dan preprocessing lulus

## Dataset yang Benar-Benar Digunakan

Analisis pada milestone ini hanya menggunakan:

- `data/raw/uci/mushroom.csv`
- `data/raw/secondary/secondary_mushroom.csv`

Tidak ada penyalinan `df_sec = df_uci.copy()` atau bentuk substitusi dataset lain.

### UCI Mushroom Dataset

- Path: `data/raw/uci/mushroom.csv`
- Shape mentah: `8124 x 23`
- Target: `class`
- Split analisis:
  - train: `6499`
  - test: `1625`

### Secondary Mushroom Dataset

- Path: `data/raw/secondary/secondary_mushroom.csv`
- Shape mentah: `61069 x 21`
- Setelah exact deduplication: `60923 x 21`
- Kolom yang di-drop karena missing `>80%` pada train split:
  - `spore-print-color`
  - `stem-root`
  - `veil-color`
  - `veil-type`
- Shape analisis setelah drop kolom: `60923 x 17`
- Split analisis:
  - train: `48738`
  - test: `12185`
- Sumber dataset untuk pelaporan: Secondary Mushroom Dataset oleh Dennis Wagner, 2020

## Metode Analisis

### Aturan Cleaning

EDA mengikuti aturan cleaning yang sama dengan [src/preprocessing.py](D:\my-kisah\yayayaya\college\machine-learning\src\preprocessing.py):

- UCI:
  - `stalk-root` dipertahankan
  - missing value kategorikal diisi `unknown`
  - tidak ada baris yang dihapus
- Secondary:
  - exact duplicate rows dihapus
  - kolom dengan missing `>80%` di-drop berdasarkan train split
  - missing kategorikal diisi `unknown`
  - missing numerik diisi median

### Anti-Data Leakage

- Split dilakukan sebelum analisis yang memengaruhi pemilihan fitur.
- Train split dipakai untuk:
  - ranking asosiasi UCI
  - Cramér's V heatmap UCI
  - mutual information ranking Secondary
  - visualisasi class-conditioned yang relevan terhadap pemilihan fitur

### Metode Statistik

#### UCI

- Distribusi kelas dan missing value mentah: seluruh raw dataset
- Asosiasi fitur kategorikal terhadap target:
  - tabel kontingensi
  - chi-square
  - Cramér's V
- Heatmap:
  - pairwise Cramér's V hanya untuk fitur kategorikal terpilih

#### Secondary

- Distribusi kelas dan missing value mentah: seluruh raw dataset
- Statistik deskriptif numerik: train split yang sudah dibersihkan
- Visual numerik per kelas:
  - boxplot `cap-diameter`
  - boxplot `stem-height`
  - boxplot `stem-width`
- Korelasi numerik:
  - hanya pada fitur numerik
- Ranking fitur campuran:
  - `mutual_info_classif`
  - numerik tetap numerik
  - kategorikal di-encode dengan `OrdinalEncoder` khusus untuk estimasi MI, bukan `LabelEncoder`

### Metode yang Sengaja Tidak Digunakan

- Tidak menggunakan `LabelEncoder` untuk fitur input EDA
- Tidak menghitung korelasi Spearman atas seluruh fitur nominal
- Tidak membuat klaim root node Decision Tree, recall model, atau feature importance model sebelum model dilatih

## Statistik Deskriptif

### UCI

- Distribusi kelas mentah:
  - edible: `4208`
  - poisonous: `3916`
- Missing value mentah:
  - `stalk-root`: `2480`

### Secondary

Statistik numerik pada train split yang sudah dibersihkan:

- `cap-diameter`
  - mean: `6.760`
  - median: `5.900`
  - std: `5.286`
  - max: `62.340`
- `stem-height`
  - mean: `6.604`
  - median: `5.960`
  - std: `3.362`
  - max: `33.250`
- `stem-width`
  - mean: `12.200`
  - median: `10.240`
  - std: `10.027`
  - max: `103.910`

Statistik numerik per kelas pada train split:

- edible:
  - `cap-diameter` mean `7.817`, median `6.730`
  - `stem-height` mean `7.042`, median `6.230`
  - `stem-width` mean `14.373`, median `12.580`
- poisonous:
  - `cap-diameter` mean `5.908`, median `5.020`
  - `stem-height` mean `6.251`, median `5.650`
  - `stem-width` mean `10.450`, median `7.740`

Korelasi numerik Secondary:

- `cap-diameter` vs `stem-width`: `0.698`
- `cap-diameter` vs `stem-height`: `0.419`
- `stem-height` vs `stem-width`: `0.431`

## Daftar Visualisasi yang Dihasilkan

### UCI

File di `reports/figures/eda/uci/`:

- `class_distribution.png`
- `missing_values_before_cleaning.png`
- `odor_vs_class.png`
- `gill_color_vs_class.png`
- `selected_cramers_v_heatmap.png`
- `feature_association_ranking.png`

### Secondary

File di `reports/figures/eda/secondary/`:

- `class_distribution.png`
- `missing_values_before_cleaning.png`
- `cap_diameter_by_class.png`
- `stem_height_by_class.png`
- `stem_width_by_class.png`
- `gill_color_vs_class.png`
- `habitat_vs_class.png`
- `season_vs_class.png`
- `mutual_information_ranking.png`

## Tabel yang Dihasilkan

File di `reports/tables/eda/`:

- `uci_class_distribution.csv`
- `uci_missing_values_before_cleaning.csv`
- `uci_odor_vs_class.csv`
- `uci_gill_color_vs_class.csv`
- `uci_feature_association_ranking.csv`
- `uci_selected_cramers_v_matrix.csv`
- `uci_eda_summary.json`
- `secondary_class_distribution.csv`
- `secondary_missing_values_before_cleaning.csv`
- `secondary_numeric_descriptive_stats.csv`
- `secondary_numeric_descriptive_by_class.csv`
- `secondary_gill_color_vs_class.csv`
- `secondary_habitat_vs_class.csv`
- `secondary_season_vs_class.csv`
- `secondary_numeric_correlation.csv`
- `secondary_mutual_information_ranking.csv`
- `secondary_eda_summary.json`

## Hasil Ranking Fitur

### UCI: Ranking Asosiasi Kategorikal terhadap Class

Metode: chi-square + Cramér's V pada train split.

Top 10:

1. `odor` - Cramér's V `0.970`
2. `spore-print-color` - `0.755`
3. `gill-color` - `0.679`
4. `ring-type` - `0.604`
5. `stalk-surface-above-ring` - `0.591`
6. `stalk-surface-below-ring` - `0.576`
7. `gill-size` - `0.532`
8. `stalk-color-above-ring` - `0.526`
9. `stalk-color-below-ring` - `0.517`
10. `bruises` - `0.503`

Fitur terpilih untuk heatmap Cramér's V:

- `odor`
- `spore-print-color`
- `gill-color`
- `ring-type`
- `stalk-surface-above-ring`
- `stalk-surface-below-ring`

### Secondary: Ranking Mutual Information

Metode: `mutual_info_classif` pada train split dengan fitur numerik dan kategorikal yang ditangani terpisah secara benar.

Top 10:

1. `stem-width` - MI `0.0639`
2. `stem-height` - `0.0444`
3. `stem-surface` - `0.0421`
4. `stem-color` - `0.0406`
5. `cap-diameter` - `0.0357`
6. `cap-color` - `0.0313`
7. `cap-surface` - `0.0302`
8. `ring-type` - `0.0287`
9. `gill-attachment` - `0.0248`
10. `cap-shape` - `0.0189`

## Interpretasi Grafik

### UCI

- `class_distribution.png`
  - Kelas edible dan poisonous relatif seimbang pada data mentah, sehingga tidak ada imbalance berat pada level dataset penuh.
- `missing_values_before_cleaning.png`
  - Missing value mentah hanya terkonsentrasi pada `stalk-root`, sesuai keputusan Milestone 1 untuk mempertahankan fitur ini dan mengimputasi `unknown`.
- `odor_vs_class.png`
  - Beberapa kategori odor sangat terpisah menurut kelas pada train split. Misalnya `a` dan `l` muncul hanya pada edible, sedangkan `f`, `s`, dan `y` hanya muncul pada poisonous di split analisis ini. Ini mendukung skor Cramér's V yang sangat tinggi, tetapi tidak sama dengan bukti kausal maupun bukti root node model.
- `gill_color_vs_class.png`
  - `gill-color` juga memperlihatkan pemisahan kelas yang kuat, meskipun tidak setajam `odor`. Beberapa warna condong kuat ke poisonous, sementara sebagian lain lebih sering edible.
- `selected_cramers_v_heatmap.png`
  - Fitur-fitur teratas UCI membentuk kelompok asosiasi kategorikal yang saling kuat. Heatmap ini dipakai untuk melihat redundansi atau kedekatan informasi, bukan untuk menyimpulkan struktur model.
- `feature_association_ranking.png`
  - `odor` jelas dominan pada analisis bivariatif, diikuti oleh `spore-print-color` dan `gill-color`. Ranking ini relevan untuk prioritas eksplorasi fitur, tetapi belum merupakan feature importance model.

### Secondary

- `class_distribution.png`
  - Secondary dataset lebih condong ke poisonous dibanding edible pada data mentah.
- `missing_values_before_cleaning.png`
  - Missing value sangat besar pada `veil-type`, `veil-color`, `stem-root`, dan `spore-print-color`, sehingga keempatnya memang wajar di-drop oleh aturan `>80%`.
- `cap_diameter_by_class.png`
  - Distribusi `cap-diameter` cenderung lebih tinggi pada edible dibanding poisonous pada train split.
- `stem_height_by_class.png`
  - `stem-height` edible juga cenderung lebih tinggi, tetapi pemisahannya lebih sedang daripada `stem-width`.
- `stem_width_by_class.png`
  - `stem-width` menunjukkan pemisahan paling jelas di antara tiga fitur numerik, konsisten dengan posisinya sebagai peringkat MI tertinggi.
- `gill_color_vs_class.png`
  - `gill-color` memperlihatkan distribusi kelas yang berbeda antar kategori, walau bukan fitur teratas absolut menurut MI.
- `habitat_vs_class.png`
  - Habitat tertentu tampak lebih dominan pada satu kelas. Misalnya pada split analisis ini, kategori `u` dan `w` muncul hanya pada edible, sedangkan `p` muncul hanya pada poisonous.
- `season_vs_class.png`
  - Distribusi kelas berubah menurut musim; kategori `a` dan `u` lebih banyak poisonous, sementara `w` relatif lebih condong edible dibanding kategori lain.
- `mutual_information_ranking.png`
  - Fitur stem mendominasi ranking MI, menunjukkan bahwa atribut morfologi batang memberi sinyal informasi yang lebih kuat dibanding banyak fitur topi atau musim pada dataset ini.

## Batasan Analisis

- Analisis ini bersifat eksploratif dan bivariatif/mutual-information, bukan bukti performa model.
- Nilai asosiasi tinggi pada UCI tidak boleh diubah menjadi klaim root node Decision Tree tanpa pelatihan model.
- Ranking mutual information Secondary bergantung pada train split dan preprocessing yang digunakan; ranking bisa bergeser pada split lain.
- Beberapa visual kategori yang langka bisa sensitif terhadap frekuensi kecil.
- Sumber referensi audit tidak dianggap ground truth; seluruh kesimpulan milestone ini dibatasi pada dataset aktual dan pipeline yang tersedia di repository.

## File Implementasi

- [src/eda.py](D:\my-kisah\yayayaya\college\machine-learning\src\eda.py)
- [src/run_eda.py](D:\my-kisah\yayayaya\college\machine-learning\src\run_eda.py)
- [tests/test_eda.py](D:\my-kisah\yayayaya\college\machine-learning\tests\test_eda.py)
- [notebooks/03_exploratory_data_analysis.ipynb](D:\my-kisah\yayayaya\college\machine-learning\notebooks\03_exploratory_data_analysis.ipynb)
- [reports/eda_section_4_2_draft.md](D:\my-kisah\yayayaya\college\machine-learning\reports\eda_section_4_2_draft.md)

## Cara Menjalankan Ulang

```powershell
python -m pip install -r requirements.txt
python src\run_eda.py
python -m pytest -q
```

## Handoff Menuju Person 3 dan Person 4

### Handoff untuk Person 3

- Gunakan ranking UCI berbasis Cramér's V dan ranking Secondary berbasis mutual information sebagai input awal eksperimen modeling.
- Jangan mengubah ranking EDA menjadi feature importance model sebelum model dilatih.
- Perhatikan bahwa Secondary dataset memiliki fitur stem yang lebih dominan menurut MI, sedangkan UCI sangat didominasi `odor`.
- Pertimbangkan evaluasi model yang membandingkan performa dengan seluruh fitur vs subset fitur teratas.

### Handoff untuk Person 4

- Gunakan [reports/eda_section_4_2_draft.md](D:\my-kisah\yayayaya\college\machine-learning\reports\eda_section_4_2_draft.md) sebagai dasar koreksi Bab 4.2.
- Pastikan laporan final menyebut:
  - Secondary Dataset bukan salinan UCI
  - sumber Secondary Dataset: Dennis Wagner, 2020
  - preprocessing menggunakan `OneHotEncoder`, bukan `LabelEncoder`
  - `stalk-root` UCI dipertahankan dan missing diimputasi `unknown`
  - tidak ada klaim root node, recall `99%`, atau feature importance model sebelum eksperimen modeling tersedia
- Lampirkan hanya grafik dan tabel yang berasal dari `reports/figures/eda/` dan `reports/tables/eda/`.
