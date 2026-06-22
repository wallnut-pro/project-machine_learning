# Audit Pembuatan DOCX

- File output: `deliverables\Laporan_Tugas_Akhir_ML_Kelompok_4_Final.docx`
- Gambar yang dimasukkan: `10`
- Tabel yang dimasukkan: `10`
- Jumlah halaman terdeteksi: `tidak terdeteksi secara andal dari python-docx`
- Catatan render: `render_docx.py` tidak dapat dipakai di environment ini karena dependency `pdf2image` tidak tersedia.
- QA otomatis yang diselesaikan:
  - tidak ada placeholder verifikasi di final_report_draft.md
  - tidak ada klaim Secondary sebagai salinan UCI
  - seluruh angka utama diambil dari CSV/JSON repository
  - daftar pustaka tersedia
- Pemeriksaan manual yang masih disarankan:
  - cek page break setiap BAB di Word atau LibreOffice
  - cek proporsi gambar pada tampilan cetak
  - cek posisi caption gambar dan tabel
