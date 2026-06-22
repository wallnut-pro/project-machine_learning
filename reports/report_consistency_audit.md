# Audit Konsistensi Laporan Lama

Dokumen ini mengaudit [LAPORAN_TUGAS_AKHIR_ML_KELOMPOK 4.docx](D:\my-kisah\yayayaya\college\machine-learning\docs\reference\LAPORAN_TUGAS_AKHIR_ML_KELOMPOK%204.docx) terhadap hasil aktual repository.

Prioritas sumber kebenaran:

1. CSV/JSON hasil eksperimen
2. Dokumentasi milestone
3. Laporan lama

## Ringkasan Temuan

| No | Kategori | Temuan pada laporan lama | Fakta aktual repository | Koreksi |
|---|---|---|---|---|
| 1 | Sumber dataset | Secondary ditulis dari Kaggle 2023 | Dokumen milestone memakai Dennis Wagner, 2020 | Ganti sumber dan beri `[PERLU VERIFIKASI SITASI]` bila bibliografi belum dicek |
| 2 | Status Secondary | Secondary dijelaskan sebagai modifikasi/ekspansi dataset primer | Repository memakai `data/raw/secondary/secondary_mushroom.csv` sebagai dataset terpisah | Hapus narasi bahwa Secondary adalah salinan UCI |
| 3 | Preprocessing fitur | Laporan lama menyebut `LabelEncoder` untuk fitur | Pipeline aktual memakai `OneHotEncoder(handle_unknown="ignore")` | Ganti seluruh narasi preprocessing fitur |
| 4 | `stalk-root` UCI | Laporan lama mengarah ke penghapusan nilai `?` | `stalk-root` dipertahankan dan missing diimputasi `unknown` | Koreksi Bab III dan 4.1 |
| 5 | EDA kategorikal | Laporan lama menyebut heatmap Spearman untuk data kategorikal | EDA aktual memakai chi-square, Cramer's V, dan mutual information | Hapus Spearman kategorikal dari laporan final |
| 6 | Klaim root node | Laporan lama memproyeksikan `odor` pasti menjadi root dari EDA | Root node hanya boleh dinyatakan setelah model dilatih; hasil aktual: UCI `odor`, Secondary `stem-width` | Pindahkan klaim ke Bab 4.3 |
| 7 | Ketergantungan `odor` | Narasi lama mengesankan `odor` satu-satunya penyebab performa | Audit menunjukkan UCI tanpa `odor` tetap sempurna | Tegaskan adanya redundansi sinyal |
| 8 | Hasil bab IV | 4.1, 4.3, 4.4, 4.5 masih placeholder | Repository sudah punya hasil lengkap | Ganti seluruh isi bab hasil |
| 9 | Angka Secondary | Laporan lama tidak mencatat deduplikasi dan drop kolom aktual | Data awal `61.069`, duplicate `146`, sisa `60.923`, drop `veil-type`, `veil-color`, `stem-root`, `spore-print-color` | Perbarui Bab III dan 4.1 |
| 10 | Split dan encoding | Ukuran split dan fitur encoded belum akurat | UCI `6499/1625`, `117`; Secondary `48738/12185`, `105` | Tambahkan tabel ringkasan preprocessing |
| 11 | Evaluasi model | Tidak ada FP/FN aktual per model | UCI semua `FP=0 FN=0`; Secondary DT baseline `FP=6 FN=6`; DT tuned `FP=8 FN=3`; RF `FP=0 FN=0` | Gunakan tabel evaluasi final |
| 12 | Tuning | Tuning lama hanya umum | Grid aktual lengkap dan tuning tidak selalu meningkatkan performa | Revisi Bab 4.4 dan 4.5 |
| 13 | Rekomendasi model | Belum berbasis bukti | Rekomendasi aktual: UCI `Decision Tree baseline`, Secondary `Random Forest baseline` | Tambahkan di Bab V |
| 14 | Rumus | Rumus Gini, Entropy, Recall tidak tampil baik pada draft lama | Rumus perlu ditulis ulang | Tulis ulang dengan format persamaan |
| 15 | Sitasi | Ada sitasi yang tidak tervalidasi dari repository | Bibliografi draft lama belum diverifikasi | Tandai `[PERLU VERIFIKASI SITASI]` |
| 16 | Istilah | Ada typo dan istilah campuran | Laporan final harus konsisten memakai edible/poisonous, One-Hot Encoding, binary mapping | Normalisasi istilah |
| 17 | Pembagian tugas | Peran pada draft lama tidak sama dengan instruksi integrasi final | Distribusi final sudah ditetapkan pada Milestone 5 | Pakai pembagian tugas baru |

## Isu Spesifik

### Dataset dan metodologi

- Secondary tidak boleh lagi disebut "modifikasi" atau "ekspansi" UCI.
- Metodologi preprocessing lama yang menyebut `LabelEncoder` bertentangan langsung dengan `preprocessor.joblib` dan Milestone 1.
- EDA lama yang menyebut korelasi Spearman untuk nominal tidak sesuai dengan implementasi aktual.

### Klaim model dan hasil

- Root node tidak boleh diklaim dari EDA.
- Tuning tidak boleh diklaim meningkatkan performa jika metrik sama atau trade-off memburuk.
- Bab evaluasi lama belum memuat nilai confusion matrix aktual.

### Formula dan sitasi

- Setelah kalimat formula Gini dan Recall di naskah lama, formula tidak muncul dengan jelas pada ekstraksi OOXML.
- Sitasi lama tidak bisa dijadikan sumber kebenaran tanpa verifikasi manual.

## Koreksi Wajib untuk Versi Final

1. Gunakan hanya dua dataset aktual di repository.
2. Nyatakan Secondary sebagai dataset terpisah.
3. Gunakan `One-Hot Encoding` untuk fitur dan `binary mapping` untuk target.
4. Nyatakan `stalk-root` dipertahankan dengan imputasi `unknown`.
5. Nyatakan angka preprocessing, split, dan evaluasi sesuai CSV/JSON aktual.
6. Nyatakan root node aktual:
   - UCI: `odor`
   - Secondary: `stem-width`
7. Nyatakan UCI tanpa `odor` tetap sempurna.
8. Nyatakan tuning tidak selalu meningkatkan performa.
9. Nyatakan model rekomendasi:
   - UCI: `Decision Tree baseline`
   - Secondary: `Random Forest baseline`

Status audit: `SELESAI`
