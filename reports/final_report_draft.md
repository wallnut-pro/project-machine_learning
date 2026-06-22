# LAPORAN TUGAS AKHIR MACHINE LEARNING

## Klasifikasi Kelayakan Konsumsi Jamur sebagai Komoditas Pertanian Berdasarkan Ciri Morfologi Menggunakan Decision Tree dan Random Forest

Draft ini disusun dari hasil aktual repository. Jika ada konflik dengan draft lama, sumber kebenaran mengikuti CSV/JSON eksperimen dan dokumentasi milestone.

## Istilah Kunci

- `edible` = `layak konsumsi` = kelas `0`
- `poisonous` = `beracun` = kelas `1`
- `false negative` = jamur beracun diprediksi layak konsumsi
- encoding target = `binary mapping`
- encoding fitur = `One-Hot Encoding`

## BAB I PENDAHULUAN

### 1.1 Latar Belakang

Jamur digunakan sebagai bahan pangan, tetapi tidak semua jamur aman dikonsumsi. Dalam praktiknya, kemiripan ciri morfologi dapat menyulitkan identifikasi manual antara jamur layak konsumsi dan jamur beracun. Artikel Wagner, Heider, dan Hattab (2021) menekankan bahwa tugas klasifikasi jamur memang menantang karena data morfologi perlu dibuat, dikurasi, dan disimulasikan secara hati-hati untuk mendukung klasifikasi yang bermakna.

Machine learning menawarkan pendekatan terukur untuk mempelajari pola dari ciri morfologi jamur. Pada penelitian ini, dua algoritma klasifikasi digunakan, yaitu Decision Tree dan Random Forest. Fokus evaluasi diarahkan pada recall untuk kelas poisonous agar risiko false negative dapat ditekan.

### 1.2 Rumusan Masalah

1. Bagaimana menerapkan preprocessing yang konsisten dan bebas data leakage pada dua dataset jamur aktual?
2. Fitur apa yang paling informatif pada masing-masing dataset berdasarkan analisis deskriptif?
3. Bagaimana performa baseline Decision Tree dan Random Forest pada kedua dataset?
4. Apakah tuning benar-benar meningkatkan performa model, khususnya recall poisonous?
5. Model mana yang paling layak direkomendasikan untuk setiap dataset?

### 1.3 Tujuan

1. Menyusun pipeline eksperimen yang dapat dijalankan ulang.
2. Menganalisis pola statistik dua dataset jamur aktual.
3. Membangun model baseline dan model tuned.
4. Membandingkan hasil baseline vs tuned untuk memilih model terbaik.

### 1.4 Manfaat

1. Menyediakan studi kasus klasifikasi yang lengkap dan reprodusibel.
2. Menunjukkan pentingnya pemilihan metrik evaluasi berbasis risiko.
3. Menjadi dasar integrasi laporan machine learning yang berbasis bukti eksperimen.

### 1.5 Batasan Penelitian

1. Penelitian hanya memakai dua dataset yang tersedia di repository.
2. Split train/test ditetapkan `80:20` dengan `random_state=42`.
3. Test set tidak dipakai untuk fitting preprocessor maupun tuning.
4. Model yang digunakan hanya Decision Tree dan Random Forest.
5. Tidak ada external validation di luar data yang tersedia.

## BAB II TINJAUAN PUSTAKA

### 2.1 Machine Learning dan Klasifikasi

Machine learning adalah pendekatan komputasional yang mempelajari pola dari data untuk membuat prediksi. Dalam klasifikasi biner, model memetakan fitur input ke salah satu dari dua label target. Pada penelitian ini, label target adalah `edible` (kelas 0) dan `poisonous` (kelas 1).

### 2.2 Jamur sebagai Komoditas Pertanian

Jamur dapat digunakan sebagai bahan pangan, tetapi tidak semua jenis aman dikonsumsi. Wagner et al. (2021) menunjukkan bahwa klasifikasi berbasis ciri morfologi memerlukan data yang dikurasi dengan baik karena kemiripan karakteristik dapat menyulitkan pembedaan jamur edible dan poisonous.

### 2.3 Decision Tree

Decision Tree adalah algoritma supervised learning yang membagi data secara rekursif berdasarkan fitur yang paling meningkatkan kemurnian node. Hasilnya dapat dibaca sebagai aturan keputusan.

### 2.4 Gini Impurity

\[
\text{Gini}(t) = 1 - \sum_{i=1}^{K} p_i^2
\]

dengan \(p_i\) menyatakan proporsi kelas ke-\(i\) pada node \(t\). Semakin kecil nilai Gini, semakin murni node tersebut.

### 2.5 Random Forest

Random Forest adalah ensemble dari banyak Decision Tree. Setiap tree dibangun dari bootstrap sample dan subset fitur acak, lalu prediksi akhir diambil melalui majority voting.

### 2.6 GridSearchCV dan Cross Validation

GridSearchCV mengevaluasi kombinasi hyperparameter dalam grid yang telah ditentukan. Pada penelitian ini dipakai `StratifiedKFold` 5-fold agar proporsi kelas tetap terjaga pada setiap fold.

### 2.7 Confusion Matrix

Confusion Matrix terdiri dari:

- `TP`: jamur beracun diprediksi beracun
- `TN`: jamur layak konsumsi diprediksi layak konsumsi
- `FP`: jamur layak konsumsi diprediksi beracun
- `FN`: jamur beracun diprediksi layak konsumsi

### 2.8 Accuracy, Precision, Recall, dan F1-Score

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

\[
\text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Recall poisonous dipakai sebagai skor utama pada tuning karena false negative adalah kesalahan paling berisiko.

### 2.9 Penelitian Terdahulu

Draft lama memuat beberapa referensi penelitian terdahulu terkait klasifikasi jamur dan algoritma pohon keputusan. Namun, tidak semua entri berhasil diverifikasi secara bibliografis. Hasil audit referensi tersebut dicatat di [reports/citation_verification.md](D:\my-kisah\yayayaya\college\machine-learning\reports\citation_verification.md), dan hanya sumber yang dapat dipertanggungjawabkan dipertahankan dalam naskah utama.

## BAB III METODOLOGI PENELITIAN

### 3.1 Sumber dan Deskripsi Dataset

| Dataset | Sumber | Data awal | Fitur input mentah | Target |
|---|---|---:|---:|---|
| UCI Mushroom Dataset | UCI Machine Learning Repository via `ucimlrepo`, dataset ID 73 | 8.124 | 22 | `class` |
| Secondary Mushroom Dataset | Wagner, Heider, dan Hattab (2021); dataset simulasi dibuat pada 2020 dan artikel ilmiahnya diterbitkan pada 2021 | 61.069 | 20 | `class` |

Path yang digunakan:

- `data/raw/uci/mushroom.csv`
- `data/raw/secondary/secondary_mushroom.csv`

### 3.2 Alur Penelitian

1. Data acquisition
2. Inspeksi awal
3. Preprocessing
4. EDA
5. Pembangunan model baseline
6. Audit metodologis baseline
7. Hyperparameter tuning
8. Evaluasi baseline vs tuned
9. Integrasi laporan

### 3.3 Data Acquisition

- UCI diunduh memakai `fetch_ucirepo(id=73)` dan disimpan ke `data/raw/uci/mushroom.csv`.
- Secondary dibaca dari file lokal di `data/raw/secondary/secondary_mushroom.csv`.

### 3.4 Preprocessing

#### UCI

- `stalk-root` dipertahankan
- missing kategorikal diisi `unknown`
- target: `e -> 0`, `p -> 1`
- split: `6499` train dan `1625` test
- fitur setelah One-Hot Encoding: `117`

#### Secondary

- `146` exact duplicate dihapus
- kolom yang di-drop:
  - `veil-type`
  - `veil-color`
  - `stem-root`
  - `spore-print-color`
- missing kategorikal lain diisi `unknown`
- missing numerik diisi median
- target: `e -> 0`, `p -> 1`
- split: `48738` train dan `12185` test
- fitur setelah One-Hot Encoding: `105`

Implementasi teknis:

- `ColumnTransformer`
- `Pipeline`
- `OneHotEncoder(handle_unknown="ignore")`
- `SimpleImputer(strategy="median")` untuk fitur numerik

### 3.5 EDA

EDA menggunakan aturan cleaning yang sama dengan preprocessing. Analisis yang memengaruhi pemilihan fitur dilakukan pada train split untuk mencegah data leakage.

Metode:

- UCI: tabel kontingensi, chi-square, Cramer's V
- Secondary: statistik deskriptif numerik, boxplot per kelas, korelasi numerik, `mutual_info_classif`

### 3.6 Pembangunan Model

Model baseline:

- `DecisionTreeClassifier(random_state=42)`
- `RandomForestClassifier(random_state=42, n_jobs=-1)`

### 3.7 Optimasi Model

Optimasi dilakukan dengan:

- `GridSearchCV`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- scoring utama: recall poisonous (`pos_label=1`)

### 3.8 Evaluasi

Metrik evaluasi:

- accuracy
- precision poisonous
- recall poisonous
- F1 poisonous
- TN, FP, FN, TP
- waktu training atau tuning
- waktu prediksi

### 3.9 Pembagian Tugas

- `Nuhan`: Bab 4.1 dan preprocessing Bab III
- `Radit`: Bab 4.2 dan EDA Bab III
- `Ibang`: Bab 4.3 dan teori Decision Tree/Random Forest
- `Farell`: Bab 4.4, 4.5, optimasi, evaluasi, dan integrasi hasil
- `Bab I`, `Bab V`, pemeriksaan sitasi, dan format akhir: dikerjakan bersama

## BAB IV HASIL DAN PEMBAHASAN

### 4.1 Pra-Pemrosesan Data

| Dataset | Shape sebelum cleaning | Shape sesudah cleaning | Duplicate dihapus | Kolom di-drop | Split train/test | Fitur setelah encoding |
|---|---|---|---:|---|---|---:|
| UCI | `8124 x 23` | `8124 x 23` | 0 | Tidak ada | `6499 / 1625` | 117 |
| Secondary | `61069 x 21` | `60923 x 17` | 146 | `veil-type`, `veil-color`, `stem-root`, `spore-print-color` | `48738 / 12185` | 105 |

Pada UCI, missing mentah hanya muncul pada `stalk-root` sebanyak `2480`. Pada Secondary, missing sangat tinggi terkonsentrasi pada empat kolom yang kemudian di-drop.

Gambar relevan:

- `Gambar 4.1` `reports/figures/eda/uci/missing_values_before_cleaning.png`
- `Gambar 4.2` `reports/figures/eda/secondary/missing_values_before_cleaning.png`

### 4.2 Analisis Statistik Deskriptif

#### UCI

Distribusi kelas mentah:

- edible: `4208`
- poisonous: `3916`

Top fitur asosiasi:

| Peringkat | Fitur | Cramer's V |
|---|---|---:|
| 1 | `odor` | 0.970148 |
| 2 | `spore-print-color` | 0.755038 |
| 3 | `gill-color` | 0.679044 |
| 4 | `ring-type` | 0.603549 |
| 5 | `stalk-surface-above-ring` | 0.591432 |

#### Secondary

Statistik numerik train split:

| Fitur | Mean | Median | Std |
|---|---:|---:|---:|
| `cap-diameter` | 6.760 | 5.900 | 5.286 |
| `stem-height` | 6.604 | 5.960 | 3.362 |
| `stem-width` | 12.200 | 10.240 | 10.027 |

Top fitur mutual information:

| Peringkat | Fitur | Mutual Information |
|---|---|---:|
| 1 | `stem-width` | 0.063919 |
| 2 | `stem-height` | 0.044431 |
| 3 | `stem-surface` | 0.042117 |
| 4 | `stem-color` | 0.040575 |
| 5 | `cap-diameter` | 0.035736 |

Interpretasi:

- UCI sangat didominasi `odor` pada analisis bivariatif.
- Secondary lebih didominasi fitur batang, terutama `stem-width`.

Gambar relevan:

- `Gambar 4.3` `reports/figures/eda/uci/odor_vs_class.png`
- `Gambar 4.4` `reports/figures/eda/secondary/stem_width_by_class.png`
- `Gambar 4.5` `reports/figures/eda/uci/selected_cramers_v_heatmap.png`

### 4.3 Pembangunan Model

Struktur model baseline:

| Dataset | Model | Fitur input | Tree depth | Leaf | Estimator | Root aktual |
|---|---|---:|---:|---:|---:|---|
| UCI | Decision Tree | 117 | 7 | 14 | - | `odor` |
| UCI | Random Forest | 117 | - | - | 100 | - |
| Secondary | Decision Tree | 105 | 25 | 206 | - | `stem-width` |
| Secondary | Random Forest | 105 | - | - | 100 | - |

Metrik baseline:

| Dataset | Model | Accuracy | Precision poisonous | Recall poisonous | F1 poisonous |
|---|---|---:|---:|---:|---:|
| UCI | Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| UCI | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Secondary | Decision Tree | 0.9990 | 0.9991 | 0.9991 | 0.9991 |
| Secondary | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Audit metodologis baseline:

- tidak ada target leakage yang terdeteksi
- overlap exact train/test setelah preprocessing = `0` untuk kedua dataset
- shuffled-label control turun mendekati acak
- UCI tanpa `odor` tetap sempurna

Jadi, `odor` adalah root aktual UCI, tetapi bukan satu-satunya penyebab performa tinggi.

### 4.4 Optimasi Model

Best hyperparameter aktual:

| Dataset | Model | Best parameter inti | Best CV recall poisonous |
|---|---|---|---:|
| UCI | Decision Tree | `class_weight="balanced"`, `min_samples_leaf=2` | 1.0000 |
| UCI | Random Forest | `n_estimators=100`, `max_features="sqrt"` | 1.0000 |
| Secondary | Decision Tree | `class_weight="balanced"`, `min_samples_leaf=1` | 0.9983 |
| Secondary | Random Forest | `n_estimators=100`, `max_features="sqrt"` | 1.0000 |

Temuan penting:

- kedua Random Forest tuned memilih konfigurasi yang identik dengan baseline
- tuning tidak selalu meningkatkan performa

### 4.5 Evaluasi dan Perbandingan Model

| Dataset | Model | Version | Accuracy | Precision poisonous | Recall poisonous | F1 poisonous | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|
| UCI | Decision Tree | baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| UCI | Decision Tree | tuned | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| UCI | Random Forest | baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| UCI | Random Forest | tuned | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Secondary | Decision Tree | baseline | 0.9990 | 0.9991 | 0.9991 | 0.9991 | 6 | 6 |
| Secondary | Decision Tree | tuned | 0.9991 | 0.9988 | 0.9996 | 0.9992 | 8 | 3 |
| Secondary | Random Forest | baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Secondary | Random Forest | tuned | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

Fokus false negative:

- UCI semua baseline dan tuned: `FN=0`
- Secondary DT baseline: `FN=6`
- Secondary DT tuned: `FN=3`
- Secondary RF baseline dan tuned: `FN=0`

Interpretasi:

- tuning Decision Tree pada Secondary menurunkan `FN`, tetapi `FP` naik dari `6` ke `8`
- tuned Decision Tree juga menjadi lebih kompleks
- Random Forest tidak memperoleh peningkatan terukur karena baseline sudah sempurna

Model rekomendasi:

1. `UCI`: `Decision Tree baseline`
2. `Secondary`: `Random Forest baseline`

Gambar relevan:

- `Gambar 4.8` `reports/figures/evaluation/secondary/decision_tree_baseline_confusion_matrix.png`
- `Gambar 4.9` `reports/figures/evaluation/secondary/decision_tree_tuned_confusion_matrix.png`
- `Gambar 4.10` `reports/figures/evaluation/secondary/false_negative_comparison.png`

## BAB V KESIMPULAN DAN SARAN

### 5.1 Kesimpulan

1. Pipeline eksperimen berhasil dibangun ulang dan dijalankan pada dua dataset aktual.
2. UCI memiliki `8124` baris mentah, `22` fitur input, split `6499/1625`, dan `117` fitur encoded; `stalk-root` dipertahankan dengan imputasi `unknown`.
3. Secondary memiliki `61069` data awal, `146` duplicate dihapus, `60923` data setelah cleaning, split `48738/12185`, dan `105` fitur encoded.
4. Root Decision Tree aktual adalah `odor` untuk UCI dan `stem-width` untuk Secondary.
5. UCI tanpa `odor` tetap sempurna, sehingga performa tinggi tidak boleh diklaim hanya bergantung pada `odor`.
6. Tuning tidak selalu meningkatkan performa. Random Forest pada kedua dataset tidak memperoleh peningkatan terukur.
7. Model rekomendasi akhir:
   - UCI: `Decision Tree baseline`
   - Secondary: `Random Forest baseline`

### 5.2 Saran

1. Pastikan daftar pustaka final tetap konsisten dengan sumber yang sudah diverifikasi dan jangan menambahkan referensi tanpa audit bibliografis.
2. Tambahkan external validation atau group-wise split jika metadata biologis tersedia.
3. Lakukan analisis probabilitas dan threshold untuk skenario berisiko tinggi.

## LAMPIRAN

### Lampiran A. Jobdesk

- `Nuhan`: Bab 4.1 dan preprocessing Bab III
- `Radit`: Bab 4.2 dan EDA Bab III
- `Ibang`: Bab 4.3 dan teori Decision Tree/Random Forest
- `Farell`: Bab 4.4, 4.5, optimasi, evaluasi, dan integrasi hasil
- `Bab I`, `Bab V`, pemeriksaan sitasi, dan format akhir`: dikerjakan bersama

### Lampiran B. Perintah Menjalankan Proyek

```powershell
python src\download_data.py
python src\run_preprocessing.py --dataset all
python src\run_eda.py
python src\run_modeling.py --dataset all
python src\run_optimization_evaluation.py --dataset all
python -m pytest -q
```

### Lampiran C. Struktur Repository

```text
machine-learning/
|-- artifacts/
|-- data/
|-- docs/
|-- notebooks/
|-- reports/
|-- src/
`-- tests/
```

### Lampiran D. Hasil Test

```text
25 passed in 17.28s
```

## DAFTAR PUSTAKA

Wagner, D., Heider, D., & Hattab, G. (2021). Mushroom data creation, curation, and simulation to support classification tasks. *Scientific Reports, 11*, 8134. https://doi.org/10.1038/s41598-021-87602-3
