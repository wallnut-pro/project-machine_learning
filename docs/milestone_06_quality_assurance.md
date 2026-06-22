# Milestone 06: Quality Assurance dan Pembuatan Laporan DOCX

## Status

Status milestone saat ini: `COMPLETED`

Status ini valid karena:

- file DOCX final berhasil dibuat
- audit pembuatan DOCX tersedia
- test generation report tersedia
- seluruh test proyek lulus

## Output

Artefak utama:

- [deliverables/Laporan_Tugas_Akhir_ML_Kelompok_4_Final.docx](D:\my-kisah\yayayaya\college\machine-learning\deliverables\Laporan_Tugas_Akhir_ML_Kelompok_4_Final.docx)
- [reports/docx_generation_audit.md](D:\my-kisah\yayayaya\college\machine-learning\reports\docx_generation_audit.md)
- [src/build_report_docx.py](D:\my-kisah\yayayaya\college\machine-learning\src\build_report_docx.py)
- [tests/test_report_generation.py](D:\my-kisah\yayayaya\college\machine-learning\tests\test_report_generation.py)

## QA yang Diselesaikan

1. Format dasar DOCX diatur ke:
   - A4
   - margin kiri 4 cm, atas 4 cm, kanan 3 cm, bawah 3 cm
   - Times New Roman 12 pt
   - spasi 1,5
   - paragraf justify
   - indentasi awal paragraf 1,25 cm
2. Cover dibuat berdasarkan identitas kelompok pada laporan lama.
3. BAB I sampai BAB V, lampiran, dan daftar pustaka tersedia.
4. Gambar terpilih dari indeks dimasukkan ke DOCX.
5. Tabel utama dari artefak eksperimen dimasukkan ke DOCX.
6. Placeholder verifikasi tidak ada di draft final yang dipakai untuk pembuatan DOCX.
7. Tidak ada klaim bahwa Secondary adalah salinan UCI.
8. Narasi tetap konsisten menyebut `One-Hot Encoding` untuk fitur.

## Keterbatasan

- Jumlah halaman tidak dapat dideteksi secara andal dari `python-docx`.
- Upaya render lewat `render_docx.py` belum berhasil di environment ini karena dependency `pdf2image` tidak tersedia.
- Karena itu, pengecekan visual akhir atas page break, proporsi gambar, dan posisi caption masih perlu dilakukan manual di Word atau LibreOffice.

## Cara Menjalankan Ulang

```powershell
python src\build_report_docx.py
python -m pytest -q
```
