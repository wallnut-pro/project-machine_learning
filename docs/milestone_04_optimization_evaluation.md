# Milestone 04: Optimization and Evaluation

## Status

Status milestone saat ini: `COMPLETED`

Status ini valid karena:

- empat tuned model berhasil dibuat
- seluruh model baseline dan tuned telah dievaluasi
- tabel dan visualisasi tersedia
- best parameter tercatat
- false negative telah diinterpretasikan
- seluruh test lulus

## Tujuan Milestone

Milestone 4 bertujuan:

- melakukan optimasi hyperparameter untuk Decision Tree dan Random Forest
- membandingkan performa baseline vs tuned pada dua dataset aktual
- memfokuskan evaluasi pada recall kelas poisonous agar false negative serendah mungkin
- menilai apakah tuning benar-benar memberi peningkatan yang terukur atau hanya mengubah konfigurasi tanpa manfaat nyata

## Input dari Milestone 1 dan 3

Milestone ini memakai artefak yang sudah tersedia, tanpa preprocessing ulang:

- `data/processed/uci/`
- `data/processed/secondary/`
- `artifacts/models/uci/`
- `artifacts/models/secondary/`

File yang digunakan:

- `X_train.npz`
- `X_test.npz`
- `y_train.npy`
- `y_test.npy`
- model baseline Decision Tree
- model baseline Random Forest

Test set tidak dipakai pada proses tuning atau pemilihan hyperparameter.

## Metode Optimasi

Optimasi dilakukan dengan `GridSearchCV` menggunakan:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `scoring=recall_poisonous` dengan `pos_label=1`
- `n_jobs=-1`
- `refit=True`

Alasan memakai recall poisonous sebagai skor utama:

- false negative berarti jamur poisonous diprediksi edible
- kesalahan ini lebih berbahaya daripada false positive
- karena itu pemilihan model diarahkan untuk meminimalkan missed poisonous cases

## Parameter yang Diuji

Grid yang dijalankan tidak dikurangi. Seluruh kombinasi sesuai instruksi.

### Decision Tree

```python
{
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"]
}
```

### Random Forest

```python
{
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"]
}
```

## Best Parameter Aktual

Tabel utama ada di [reports/tables/evaluation/best_hyperparameters.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\evaluation\best_hyperparameters.csv).

### UCI Decision Tree

- `class_weight="balanced"`
- `max_depth=None`
- `min_samples_leaf=2`
- `min_samples_split=2`
- best CV recall poisonous: `1.0000`
- root feature aktual: `odor`

### UCI Random Forest

- `class_weight=None`
- `max_depth=None`
- `max_features="sqrt"`
- `min_samples_split=2`
- `n_estimators=100`
- best CV recall poisonous: `1.0000`

### Secondary Decision Tree

- `class_weight="balanced"`
- `max_depth=None`
- `min_samples_leaf=1`
- `min_samples_split=2`
- best CV recall poisonous: `0.9983`
- root feature aktual: `stem-width`

### Secondary Random Forest

- `class_weight=None`
- `max_depth=None`
- `max_features="sqrt"`
- `min_samples_split=2`
- `n_estimators=100`
- best CV recall poisonous: `1.0000`

Catatan penting:

- kedua Random Forest tuned memilih konfigurasi yang identik dengan baseline
- artinya tuning tidak memberi peningkatan terukur pada model ini

## Hasil Evaluasi Delapan Model

Tabel perbandingan utama ada di [reports/tables/evaluation/final_model_comparison.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\evaluation\final_model_comparison.csv).

### UCI

- Decision Tree baseline:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`
- Decision Tree tuned:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`
- Random Forest baseline:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`
- Random Forest tuned:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`

Interpretasi:

- tuning tidak meningkatkan metrik UCI sama sekali
- tuned Decision Tree sedikit berubah strukturnya:
  - depth `7 -> 6`
  - leaf `14 -> 19`
- karena skor identik, klaim peningkatan performa tidak valid

### Secondary

- Decision Tree baseline:
  - accuracy `0.9990`
  - precision poisonous `0.9991`
  - recall poisonous `0.9991`
  - F1 poisonous `0.9991`
  - FP `6`
  - FN `6`
- Decision Tree tuned:
  - accuracy `0.9991`
  - precision poisonous `0.9988`
  - recall poisonous `0.9996`
  - F1 poisonous `0.9992`
  - FP `8`
  - FN `3`
- Random Forest baseline:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`
- Random Forest tuned:
  - accuracy `1.0000`
  - precision poisonous `1.0000`
  - recall poisonous `1.0000`
  - F1 poisonous `1.0000`
  - FP `0`
  - FN `0`

Interpretasi:

- tuned Decision Tree memang meningkatkan recall poisonous
- false negative turun dari `6` menjadi `3`
- tetapi false positive naik dari `6` menjadi `8`
- model juga menjadi lebih kompleks:
  - depth `25 -> 32`
  - leaf `206 -> 247`
- tuned Random Forest tidak memberi peningkatan sama sekali karena baseline sudah sempurna dan parameter terbaik kembali ke konfigurasi baseline

## Interpretasi Confusion Matrix

Tabel nilai confusion matrix ada di [reports/tables/evaluation/confusion_matrix_values.csv](D:\my-kisah\yayayaya\college\machine-learning\reports\tables\evaluation\confusion_matrix_values.csv).

Fokus utama evaluasi adalah false negative:

- `FN` berarti jamur poisonous diprediksi edible
- ini adalah kesalahan paling berisiko

Hasil aktual:

- UCI semua model: `FN=0`
- Secondary Random Forest baseline dan tuned: `FN=0`
- Secondary Decision Tree baseline: `FN=6`
- Secondary Decision Tree tuned: `FN=3`

Kesimpulan:

- pada Secondary, tuning Decision Tree membantu menurunkan FN, tetapi belum mengungguli Random Forest yang sudah mencapai `FN=0`
- pada UCI, semua model sudah nol FN sehingga tuning tidak membawa manfaat praktis tambahan

## Perbandingan Kompleksitas Model

### UCI Decision Tree

- baseline:
  - depth `7`
  - leaf `14`
- tuned:
  - depth `6`
  - leaf `19`

Interpretasi:

- tuned tree sedikit lebih dangkal tetapi memiliki leaf lebih banyak
- karena metrik sama persis, baseline tetap lebih hemat waktu training dan cukup untuk dipilih

### Secondary Decision Tree

- baseline:
  - depth `25`
  - leaf `206`
- tuned:
  - depth `32`
  - leaf `247`

Interpretasi:

- tuned tree lebih kompleks
- ada perbaikan FN, tetapi trade-off terhadap kompleksitas dan FP harus dicatat

### Random Forest

- UCI baseline dan tuned:
  - `n_estimators=100`
- Secondary baseline dan tuned:
  - `n_estimators=100`

Interpretasi:

- tuning tidak mengubah struktur ensemble Random Forest pada kedua dataset

## Model Terbaik per Dataset

### UCI

Model terbaik untuk penggunaan praktis: `Decision Tree baseline`

Alasan:

- seluruh model mencapai skor sempurna
- `FN=0` dan `FP=0`
- training time baseline paling kecil
- struktur pohon tetap paling mudah dijelaskan tanpa kehilangan performa

### Secondary

Model terbaik untuk penggunaan praktis: `Random Forest baseline`

Alasan:

- sudah mencapai accuracy, precision, recall, dan F1 = `1.0000`
- `FN=0` dan `FP=0`
- tuned Random Forest memilih parameter yang sama, jadi tuning tidak memberi keuntungan tambahan
- tuned Decision Tree memang menurunkan FN, tetapi masih kalah dari Random Forest yang sudah nol FN sejak baseline

## Apakah Tuning Memberikan Peningkatan?

Jawaban singkat:

- `UCI`: tidak
- `Secondary Decision Tree`: ya, tetapi kecil dan disertai trade-off
- `Secondary Random Forest`: tidak

Penjelasan:

- Pada UCI, tuning tidak meningkatkan metrik karena semua model sudah sempurna.
- Pada Secondary Decision Tree, tuning meningkatkan recall poisonous dan menurunkan FN `6 -> 3`, tetapi menaikkan FP `6 -> 8` dan membuat model lebih kompleks.
- Pada kedua Random Forest, tuning tidak meningkatkan performa karena konfigurasi terbaik sama dengan baseline.

Karena baseline Random Forest sudah sempurna, optimasi hanya mengonfirmasi bahwa konfigurasi baseline tersebut sudah cukup baik untuk split yang tersedia.

## Artefak yang Dihasilkan

### Model Tuned

- `artifacts/models/uci/decision_tree_tuned.joblib`
- `artifacts/models/uci/random_forest_tuned.joblib`
- `artifacts/models/secondary/decision_tree_tuned.joblib`
- `artifacts/models/secondary/random_forest_tuned.joblib`

### Grid Search

- `artifacts/grid_search/uci/best_params.json`
- `artifacts/grid_search/uci/cv_results.csv`
- `artifacts/grid_search/secondary/best_params.json`
- `artifacts/grid_search/secondary/cv_results.csv`

### Tabel Evaluasi

- `reports/tables/evaluation/final_model_comparison.csv`
- `reports/tables/evaluation/best_hyperparameters.csv`
- `reports/tables/evaluation/confusion_matrix_values.csv`
- `reports/tables/evaluation/optimization_summary.json`

### Visualisasi

Untuk masing-masing dataset (`uci` dan `secondary`) tersedia:

- confusion matrix baseline Decision Tree
- confusion matrix tuned Decision Tree
- confusion matrix baseline Random Forest
- confusion matrix tuned Random Forest
- grafik comparison accuracy
- grafik comparison precision poisonous
- grafik comparison recall poisonous
- grafik comparison F1 poisonous
- grafik comparison false negative
- grafik comparison training/tuning time

## Keterbatasan

- Tuning tetap memakai satu train/test split akhir; test set tidak dipakai untuk tuning, tetapi hasil akhir tetap bergantung pada split ini.
- Secondary dataset tidak memiliki group/species identifier eksplisit untuk audit group-wise generalization.
- Skor sempurna pada UCI dan Random Forest Secondary tidak otomatis berarti model akan sama kuat di data luar distribusi.
- Belum ada kalibrasi probabilitas, threshold analysis, atau external validation.

## Cara Menjalankan Ulang

```powershell
python src\run_optimization_evaluation.py --dataset all
python -m pytest -q
```

## Handoff untuk Integrasi Laporan

### Untuk penulis Bab 4.4

- gunakan best parameter dari `best_hyperparameters.csv`
- tulis bahwa recall poisonous dipakai sebagai scoring utama karena FN lebih berbahaya
- jangan klaim semua tuning meningkatkan performa; pada beberapa kasus tuning hanya mempertahankan hasil baseline

### Untuk penulis Bab 4.5

- gunakan `final_model_comparison.csv` dan `confusion_matrix_values.csv`
- fokuskan pembahasan pada FN, terutama Secondary Decision Tree vs Random Forest
- jelaskan bahwa feature importance dan root node dibahas di Milestone 3, sedangkan Bab 4.5 fokus pada evaluasi hasil tuning

### Untuk integrasi kesimpulan akhir

- UCI: baseline Decision Tree sudah cukup karena semua model identik secara metrik
- Secondary: baseline Random Forest tetap pilihan terbaik
- tuning berfungsi lebih sebagai verifikasi robustness daripada peningkatan universal
