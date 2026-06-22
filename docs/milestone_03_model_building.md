# Milestone 03: Model Building

## Status

Status milestone saat ini: `COMPLETED`

Status ini tetap valid setelah sanity-check metodologis karena:

- empat model baseline berhasil dilatih
- seluruh artefak model dan prediksi tersedia
- grafik dan tabel utama tersedia
- root node aktual Decision Tree tercatat
- tidak ada indikasi target leakage
- shuffled-label control turun ke performa mendekati acak
- exact feature-vector overlap train/test setelah preprocessing adalah `0` untuk kedua dataset
- cross-validation pada training set konsisten
- seluruh test lulus

## Tujuan Milestone

Milestone 3 bertujuan membangun baseline model klasifikasi untuk dua dataset jamur menggunakan:

- Decision Tree
- Random Forest

Model dilatih langsung dari output preprocessing Milestone 1, tanpa preprocessing ulang dan tanpa menggunakan test set pada proses `fit`.

## Input dari Milestone 1 dan 2

Input model building berasal dari artefak preprocessing:

- `data/processed/uci/`
- `data/processed/secondary/`

File yang dipakai:

- `X_train.npz`
- `X_test.npz`
- `y_train.npy`
- `y_test.npy`
- `feature_names.csv`
- `preprocessor.joblib`

Temuan EDA Milestone 2 dipakai hanya sebagai bahan interpretasi, bukan sebagai sumber asumsi struktur model.

## Konfigurasi Model Baseline

### Decision Tree

- `DecisionTreeClassifier(random_state=42)`

### Random Forest

- `RandomForestClassifier(random_state=42, n_jobs=-1)`

Catatan:

- Tidak ada hyperparameter tuning pada milestone ini.
- Seluruh parameter lain tetap default sesuai instruksi.

## Cara Kerja Decision Tree

Decision Tree membagi data secara rekursif berdasarkan fitur yang paling menurunkan impurity pada setiap node. Pada baseline ini, pemisahan dilakukan pada fitur hasil One-Hot Encoding dari Milestone 1.

### Gini Impurity dan Proses Split

Untuk setiap kandidat split, model mengevaluasi impurity node dengan Gini:

- node murni memiliki impurity rendah
- node dengan campuran kelas memiliki impurity lebih tinggi

Model memilih split yang memberikan penurunan impurity terbesar. Root node aktual pada milestone ini diambil langsung dari model yang telah dilatih, bukan diasumsikan dari EDA.

## Cara Kerja Random Forest

Random Forest membangun banyak Decision Tree lalu menggabungkan prediksinya.

Komponen penting:

- bootstrap sampling:
  - setiap tree dilatih pada sampel bootstrap dari train set
- feature randomness:
  - setiap split hanya melihat subset fitur acak
- majority voting:
  - prediksi akhir klasifikasi ditentukan dari suara mayoritas semua tree

Pendekatan ini biasanya mengurangi overfitting dibanding satu pohon tunggal, meskipun hasil aktual tetap harus dibaca dari metrik yang diperoleh.

## Hasil Struktur Model

Tabel ringkasan struktur tersimpan di [reports/tables/modeling/model_structure_summary.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\model_structure_summary.csv).

### UCI

- Decision Tree:
  - jumlah fitur input: `117`
  - depth: `7`
  - jumlah leaf: `14`
  - encoded root feature: `categorical__odor_n`
  - original root feature: `odor`
- Random Forest:
  - jumlah fitur input: `117`
  - jumlah estimator: `100`

### Secondary

- Decision Tree:
  - jumlah fitur input: `105`
  - depth: `25`
  - jumlah leaf: `206`
  - encoded root feature: `numeric__stem-width`
  - original root feature: `stem-width`
- Random Forest:
  - jumlah fitur input: `105`
  - jumlah estimator: `100`

## Root Node Aktual

Root node Decision Tree yang benar-benar dilatih:

- UCI Decision Tree:
  - encoded root feature: `categorical__odor_n`
  - original root feature: `odor`
- Secondary Decision Tree:
  - encoded root feature: `numeric__stem-width`
  - original root feature: `stem-width`

Perbandingan dengan EDA:

- UCI:
  - EDA menempatkan `odor` sebagai fitur asosiasi terkuat terhadap target.
  - Model baseline juga memilih `odor` sebagai root node aktual.
  - Ini adalah hasil yang terkonfirmasi oleh training model, bukan asumsi.
- Secondary:
  - EDA menempatkan `stem-width` sebagai fitur mutual information tertinggi.
  - Model baseline juga memilih `stem-width` sebagai root node aktual.
  - Ini konsisten dengan sinyal numerik yang terlihat pada EDA.

Catatan penting:

- Secondary tidak memiliki fitur `odor`, sehingga tidak ada pembahasan `odor` untuk dataset ini.

## Feature Importance Aktual

Tabel feature importance ada di:

- [reports/tables/modeling/feature_importance_uci.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\feature_importance_uci.csv)
- [reports/tables/modeling/feature_importance_secondary.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\feature_importance_secondary.csv)

### UCI Random Forest

Top encoded feature importance:

1. `categorical__odor_n` - `0.1413`
2. `categorical__gill-size_n` - `0.0666`
3. `categorical__odor_f` - `0.0580`
4. `categorical__gill-size_b` - `0.0528`
5. `categorical__spore-print-color_h` - `0.0527`

Top aggregated original feature importance:

1. `odor` - `0.2462`
2. `gill-size` - `0.1194`
3. `spore-print-color` - `0.0894`
4. `stalk-surface-below-ring` - `0.0764`
5. `ring-type` - `0.0662`

### Secondary Random Forest

Top encoded feature importance:

1. `numeric__stem-width` - `0.0865`
2. `numeric__stem-height` - `0.0579`
3. `numeric__cap-diameter` - `0.0544`
4. `categorical__stem-color_w` - `0.0356`
5. `categorical__gill-spacing_d` - `0.0266`

Top aggregated original feature importance:

1. `cap-surface` - `0.1156`
2. `gill-attachment` - `0.1070`
3. `stem-width` - `0.0865`
4. `gill-color` - `0.0854`
5. `stem-color` - `0.0807`

Interpretasi:

- Feature importance menunjukkan kontribusi fitur terhadap keputusan model pada baseline ini.
- Nilai importance bukan hubungan sebab-akibat.
- Importance juga tidak identik dengan asosiasi statistik dari EDA; karena itu hasil Random Forest perlu dibaca sebagai perilaku model, bukan sebagai bukti kausal.

## Metrik Baseline

Tabel metrik ada di [reports/tables/modeling/baseline_metrics.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\baseline_metrics.csv).

Kelas positif untuk precision, recall, dan F1:

- `poisonous`
- `pos_label=1`

### Hasil

- Secondary Decision Tree:
  - train accuracy: `1.0000`
  - test accuracy: `0.9990`
  - precision poisonous: `0.9991`
  - recall poisonous: `0.9991`
  - f1 poisonous: `0.9991`
- Secondary Random Forest:
  - train accuracy: `1.0000`
  - test accuracy: `1.0000`
  - precision poisonous: `1.0000`
  - recall poisonous: `1.0000`
  - f1 poisonous: `1.0000`
- UCI Decision Tree:
  - train accuracy: `1.0000`
  - test accuracy: `1.0000`
  - precision poisonous: `1.0000`
  - recall poisonous: `1.0000`
  - f1 poisonous: `1.0000`
- UCI Random Forest:
  - train accuracy: `1.0000`
  - test accuracy: `1.0000`
  - precision poisonous: `1.0000`
  - recall poisonous: `1.0000`
  - f1 poisonous: `1.0000`

## Sanity-Check Metodologis

Tabel audit ada di:

- [reports/tables/modeling/sanity_check_results.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\sanity_check_results.csv)
- [reports/tables/modeling/cross_validation_baseline.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\modeling\cross_validation_baseline.csv)

### 1. Cek Target Leakage

Hasil:

- `target_in_preprocessor_inputs = False` untuk semua model
- `target_in_feature_names = False` untuk semua model
- `fit_train_only = True` untuk semua model

Interpretasi:

- Kolom target `class` tidak masuk ke `X` maupun `feature_names`.
- Model hanya di-fit menggunakan `X_train` dan `y_train`.

### 2. Exact Feature-Vector Overlap antara Train dan Test

Hasil untuk kedua dataset:

- `exact_overlap_unique_feature_vectors = 0`
- `overlapping_test_rows_feature_vectors = 0`
- `exact_overlap_unique_feature_label_patterns = 0`
- `overlapping_test_rows_feature_label_patterns = 0`

Interpretasi:

- Tidak ada vektor fitur identik yang muncul di train dan test setelah preprocessing.
- Tidak ada pola fitur+label identik lintas train-test yang dapat langsung menjelaskan skor hampir sempurna.

### 3. Shuffled-Label Control

Hasil:

- Secondary Decision Tree:
  - accuracy `0.5158`
  - recall poisonous `0.5660`
  - F1 poisonous `0.5643`
- Secondary Random Forest:
  - accuracy `0.5157`
  - recall poisonous `0.6047`
  - F1 poisonous `0.5804`
- UCI Decision Tree:
  - accuracy `0.5212`
  - recall poisonous `0.5006`
  - F1 poisonous `0.5019`
- UCI Random Forest:
  - accuracy `0.4917`
  - recall poisonous `0.4151`
  - F1 poisonous `0.4404`

Interpretasi:

- Setelah label train diacak, performa turun mendekati tebakan acak.
- Ini konsisten dengan tidak adanya target leakage eksplisit pada baseline utama.

### 4. StratifiedKFold 5-Fold pada Training Set

Hasil:

- Secondary Decision Tree:
  - accuracy mean/std: `0.9982 ± 0.0006`
  - recall poisonous mean/std: `0.9982 ± 0.0007`
  - F1 poisonous mean/std: `0.9984 ± 0.0006`
- Secondary Random Forest:
  - accuracy mean/std: `1.0000 ± 0.0000`
  - recall poisonous mean/std: `1.0000 ± 0.0000`
  - F1 poisonous mean/std: `1.0000 ± 0.0000`
- UCI Decision Tree:
  - accuracy mean/std: `0.9997 ± 0.0006`
  - recall poisonous mean/std: `0.9994 ± 0.0013`
  - F1 poisonous mean/std: `0.9997 ± 0.0006`
- UCI Random Forest:
  - accuracy mean/std: `1.0000 ± 0.0000`
  - recall poisonous mean/std: `1.0000 ± 0.0000`
  - F1 poisonous mean/std: `1.0000 ± 0.0000`

Interpretasi:

- Performa pada training folds konsisten dengan hasil test split.
- Tidak terlihat gejala bahwa skor tinggi hanya muncul pada satu split kebetulan.

### 5. Confusion Matrix Numerik

Hasil aktual pada test set:

- Secondary Decision Tree:
  - TN `5430`
  - FP `6`
  - FN `6`
  - TP `6743`
- Secondary Random Forest:
  - TN `5436`
  - FP `0`
  - FN `0`
  - TP `6749`
- UCI Decision Tree:
  - TN `842`
  - FP `0`
  - FN `0`
  - TP `783`
- UCI Random Forest:
  - TN `842`
  - FP `0`
  - FN `0`
  - TP `783`

### 6. Eksperimen Diagnostik UCI tanpa Fitur `odor`

Eksperimen ini hanya untuk analisis tambahan dan tidak mengganti model utama.

Hasil:

- jumlah encoded feature `odor` yang dihapus: `9`
- UCI Decision Tree tanpa `odor`:
  - test accuracy `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
- UCI Random Forest tanpa `odor`:
  - test accuracy `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`

Interpretasi:

- Skor tidak turun ketika fitur `odor` dihapus.
- Ini menunjukkan bahwa UCI memiliki redundansi fitur yang sangat tinggi; `odor` penting, tetapi bukan satu-satunya sinyal yang mampu memisahkan kelas.

### 7. Keterbatasan Tambahan untuk Secondary

- Pada artefak processed yang tersedia, tidak ada species/group identifier eksplisit.
- Exact overlap train-test setelah preprocessing memang nol.
- Namun, jika data berasal dari proses simulasi atau generasi terstruktur, sampel yang sangat mirip masih mungkin tersebar antara train dan test tanpa bisa diverifikasi di level group.
- Ini harus ditulis sebagai keterbatasan penelitian, bukan sebagai bukti leakage.

## Visualisasi yang Dihasilkan

### Decision Tree

- [reports/figures/modeling/uci/decision_tree_top_levels.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\uci\decision_tree_top_levels.png)
- [reports/figures/modeling/secondary/decision_tree_top_levels.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\secondary\decision_tree_top_levels.png)

Plot ini dibatasi `max_depth=3` agar struktur top-level tetap terbaca.

### Random Forest Feature Importance

#### UCI

- [reports/figures/modeling/uci/random_forest_feature_importance_top20_encoded.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\uci\random_forest_feature_importance_top20_encoded.png)
- [reports/figures/modeling/uci/random_forest_feature_importance_top20_aggregated.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\uci\random_forest_feature_importance_top20_aggregated.png)

#### Secondary

- [reports/figures/modeling/secondary/random_forest_feature_importance_top20_encoded.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\secondary\random_forest_feature_importance_top20_encoded.png)
- [reports/figures/modeling/secondary/random_forest_feature_importance_top20_aggregated.png](D:\my-kisah\yayayaya\college\machine-learning\reports\figures\modeling\secondary\random_forest_feature_importance_top20_aggregated.png)

## Artefak Model dan Prediksi

### Model

- `artifacts/models/uci/decision_tree_baseline.joblib`
- `artifacts/models/uci/random_forest_baseline.joblib`
- `artifacts/models/secondary/decision_tree_baseline.joblib`
- `artifacts/models/secondary/random_forest_baseline.joblib`

### Prediksi

- `artifacts/predictions/uci/decision_tree_y_pred.npy`
- `artifacts/predictions/uci/random_forest_y_pred.npy`
- `artifacts/predictions/secondary/decision_tree_y_pred.npy`
- `artifacts/predictions/secondary/random_forest_y_pred.npy`

## Apakah Skor Sempurna Dapat Dipercaya?

Jawaban singkat: skor sangat tinggi ini dapat dipercaya sebagai hasil evaluasi pada split yang tersedia, tetapi tetap harus dibaca dengan hati-hati.

Alasan yang mendukung kepercayaan:

- tidak ada target leakage yang terdeteksi
- shuffled-label control turun ke kisaran acak
- exact overlap train-test setelah preprocessing adalah nol
- cross-validation pada training set konsisten sangat tinggi

Alasan untuk tetap berhati-hati:

- UCI tetap sempurna bahkan tanpa `odor`, yang berarti dataset memiliki redundansi sinyal yang sangat kuat
- Secondary tidak memiliki group/species identifier, sehingga kemiripan terstruktur antar sampel tidak bisa diuji pada level grup
- baseline ini belum menguji generalisasi di luar skema split yang tersedia

Kesimpulan praktis:

- hasil baseline bukan artefak leakage yang jelas
- tetapi generalisasi nyata tetap harus diuji lagi pada Milestone 4 melalui tuning, validasi tambahan, dan analisis robustnes

## Keterbatasan Model Baseline

- Model masih baseline default, belum ada tuning hyperparameter.
- Feature importance tidak sama dengan sebab-akibat biologis.
- Hasil ini belum membandingkan robustnes antar skenario tuning atau validasi tambahan.
- Baseline ini belum menguji trade-off interpretabilitas vs generalisasi secara sistematis.
- Secondary tidak memiliki group identifier untuk menguji group-wise split.

## Handoff untuk Milestone 4

Fokus Milestone 4:

- hyperparameter tuning untuk Decision Tree dan Random Forest
- validasi apakah performa sangat tinggi tetap stabil setelah tuning protocol yang lebih ketat
- analisis overfitting dan generalisasi
- perbandingan model tuned vs baseline
- evaluasi apakah root split atau ranking importance berubah setelah tuning

Yang perlu dibawa ke Milestone 4:

- root node aktual Decision Tree dari baseline
- feature importance baseline Random Forest
- metrik baseline per dataset
- hasil sanity-check metodologis
- artefak model dan prediksi untuk pembanding

## Cara Menjalankan Ulang

```powershell
python src\run_modeling.py --dataset all
python -m pytest -q
```
