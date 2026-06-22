# Milestone 05: Report Integration

## Status

Status milestone saat ini: `COMPLETED`

Status ini valid karena:

- laporan terintegrasi tersedia
- audit konsistensi selesai
- indeks gambar dan tabel tersedia
- angka hasil sudah diselaraskan dengan repository
- seluruh test tetap lulus

## Tujuan

Milestone 5 mengintegrasikan hasil Milestone 1 sampai 4 ke dalam satu draft laporan yang konsisten secara metodologis dan numerik, sambil mengaudit draft lama agar klaim yang salah tidak terbawa ke versi final.

## Sumber Utama

Prioritas sumber:

1. CSV/JSON hasil eksperimen pada `reports/tables/` dan `data/processed/*/preprocessing_summary.json`
2. Dokumentasi milestone 1 sampai 4
3. Draft lama:
   - [LAPORAN_TUGAS_AKHIR_ML_KELOMPOK 4.docx](D:\my-kisah\yayayaya\college\machine-learning\docs\reference\LAPORAN_TUGAS_AKHIR_ML_KELOMPOK%204.docx)

## Output

Artefak yang dihasilkan:

- [reports/final_report_draft.md](D:\my-kisah\yayayaya\college\machine-learning\reports\final_report_draft.md)
- [reports/report_consistency_audit.md](D:\my-kisah\yayayaya\college\machine-learning\reports\report_consistency_audit.md)
- [reports/report_figures_and_tables_index.md](D:\my-kisah\yayayaya\college\machine-learning\reports\report_figures_and_tables_index.md)
- [docs/milestone_05_report_integration.md](D:\my-kisah\yayayaya\college\machine-learning\docs\milestone_05_report_integration.md)

Catatan DOCX:

- `python-docx` tidak tersedia di environment saat milestone ini dikerjakan.
- Karena itu, `deliverables/Laporan_Tugas_Akhir_ML_Kelompok_4_Integrated.docx` tidak dibuat.
- File laporan asli tidak diubah.

## Koreksi Kunci

1. Secondary Dataset tidak lagi diperlakukan sebagai salinan atau modifikasi UCI.
2. Narasi preprocessing diubah menjadi:
   - target: `binary mapping`
   - fitur: `One-Hot Encoding`
3. `stalk-root` UCI dinyatakan dipertahankan dan diimputasi `unknown`.
4. Root node dan hasil evaluasi hanya ditulis berdasarkan model yang benar-benar dilatih.
5. Hasil tuning ditulis sesuai tabel aktual dan tidak dilebih-lebihkan.
6. Istilah dinormalisasi:
   - edible = layak konsumsi = kelas 0
   - poisonous = beracun = kelas 1
   - false negative = jamur beracun diprediksi layak konsumsi

## Validasi Angka

Angka berikut sudah dicocokkan dengan artefak repository:

- UCI: `8124` baris mentah, `22` fitur input, split `6499/1625`, `117` fitur encoded
- Secondary: `61069` data awal, `146` duplicate dihapus, `60923` data setelah cleaning, split `48738/12185`, `105` fitur encoded
- Root Decision Tree:
  - UCI: `odor`
  - Secondary: `stem-width`
- Evaluasi:
  - UCI semua baseline dan tuned `FP=0`, `FN=0`
  - Secondary DT baseline `FP=6`, `FN=6`
  - Secondary DT tuned `FP=8`, `FN=3`
  - Secondary RF baseline dan tuned `FP=0`, `FN=0`

## Validasi Test

Perintah validasi:

```powershell
python -m pytest -q
```

Hasil test terakhir:

```text
25 passed in 17.28s
```
