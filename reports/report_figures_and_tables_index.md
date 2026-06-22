# Indeks Gambar dan Tabel Laporan Final

Dokumen ini memilih artefak paling penting dari `reports/figures/` dan `reports/tables/` untuk dimasukkan ke laporan final.

## Gambar

| No. | Judul | Path | Bagian | Interpretasi singkat |
|---|---|---|---|---|
| Gambar 4.1 | Missing value sebelum cleaning UCI | `reports/figures/eda/uci/missing_values_before_cleaning.png` | 4.1 | Menjelaskan mengapa `stalk-root` dipertahankan dan diimputasi. |
| Gambar 4.2 | Missing value sebelum cleaning Secondary | `reports/figures/eda/secondary/missing_values_before_cleaning.png` | 4.1 | Menjelaskan drop kolom dengan missing sangat tinggi. |
| Gambar 4.3 | `odor` vs class pada UCI | `reports/figures/eda/uci/odor_vs_class.png` | 4.2 | Menunjukkan asosiasi kuat `odor` terhadap target. |
| Gambar 4.4 | `stem-width` berdasarkan class pada Secondary | `reports/figures/eda/secondary/stem_width_by_class.png` | 4.2 | Menunjukkan pemisahan kelas pada fitur batang. |
| Gambar 4.5 | Heatmap Cramer's V UCI | `reports/figures/eda/uci/selected_cramers_v_heatmap.png` | 4.2 | Menunjukkan kedekatan asosiasi antar fitur kategorikal terpilih. |
| Gambar 4.6 | Top-level Decision Tree baseline UCI | `reports/figures/modeling/uci/decision_tree_top_levels.png` | 4.3 | Bukti visual struktur model dan root aktual. |
| Gambar 4.7 | Random Forest feature importance agregat Secondary | `reports/figures/modeling/secondary/random_forest_feature_importance_top20_aggregated.png` | 4.3 | Menunjukkan kontribusi fitur asli pada ensemble Secondary. |
| Gambar 4.8 | Confusion matrix DT baseline Secondary | `reports/figures/evaluation/secondary/decision_tree_baseline_confusion_matrix.png` | 4.5 | Menunjukkan `FP=6` dan `FN=6`. |
| Gambar 4.9 | Confusion matrix DT tuned Secondary | `reports/figures/evaluation/secondary/decision_tree_tuned_confusion_matrix.png` | 4.5 | Menunjukkan `FN` turun menjadi `3`, tetapi `FP` naik menjadi `8`. |
| Gambar 4.10 | Perbandingan false negative Secondary | `reports/figures/evaluation/secondary/false_negative_comparison.png` | 4.5 | Menjadi visual utama untuk diskusi risiko model. |

## Tabel

| No. | Judul | Path | Bagian | Fungsi |
|---|---|---|---|---|
| Tabel 3.1 | Ringkasan dua dataset | `data/processed/*/preprocessing_summary.json` | 3.1 | Sumber ukuran data, cleaning, dan split. |
| Tabel 4.1 | Ringkasan hasil preprocessing | `data/processed/*/preprocessing_summary.json` | 4.1 | Menjelaskan deduplikasi, drop kolom, dan jumlah fitur encoded. |
| Tabel 4.2 | Ranking asosiasi UCI | `reports/tables/eda/uci_feature_association_ranking.csv` | 4.2 | Menunjukkan dominasi `odor`. |
| Tabel 4.3 | Ranking mutual information Secondary | `reports/tables/eda/secondary_mutual_information_ranking.csv` | 4.2 | Menunjukkan dominasi fitur batang. |
| Tabel 4.4 | Struktur model baseline | `reports/tables/modeling/model_structure_summary.csv` | 4.3 | Sumber depth, leaf, estimator, dan root node aktual. |
| Tabel 4.5 | Metrik baseline | `reports/tables/modeling/baseline_metrics.csv` | 4.3 | Dasar pembanding sebelum tuning. |
| Tabel 4.6 | Audit metodologis baseline | `reports/tables/modeling/sanity_check_results.csv` dan `reports/tables/modeling/cross_validation_baseline.csv` | 4.3 | Membuktikan tidak ada leakage yang jelas. |
| Tabel 4.7 | Best hyperparameter | `reports/tables/evaluation/best_hyperparameters.csv` | 4.4 | Menunjukkan hasil tuning aktual. |
| Tabel 4.8 | Perbandingan final baseline vs tuned | `reports/tables/evaluation/final_model_comparison.csv` | 4.5 | Tabel evaluasi utama. |
| Tabel 4.9 | Nilai confusion matrix | `reports/tables/evaluation/confusion_matrix_values.csv` | 4.5 | Sumber FP/FN/TN/TP aktual. |

## Catatan

- Seluruh path di atas sudah ada di repository.
- Tidak semua gambar dimasukkan agar narasi tetap fokus.
