# Draft Koreksi Bab 4.2

## Koreksi Wajib terhadap Naskah Lama

Bagian 4.2 harus dikoreksi dengan poin berikut:

1. Secondary Dataset bukan salinan UCI. Analisis milestone ini menggunakan file aktual `data/raw/secondary/secondary_mushroom.csv`, bukan hasil `df_sec = df_uci.copy()`.
2. Sumber Secondary Dataset pada laporan harus diperbaiki menjadi Dennis Wagner, 2020.
3. Preprocessing yang dipakai pada pipeline proyek adalah `OneHotEncoder(handle_unknown="ignore")`, bukan `LabelEncoder` untuk fitur input.
4. Pada UCI, fitur `stalk-root` dipertahankan dan missing value diimputasi sebagai `unknown`; fitur ini tidak dibuang.
5. Jangan mengklaim `odor` sebagai root node Decision Tree atau recall `99%` sebelum model benar-benar dilatih dan dievaluasi.

## Draf Narasi Pengganti Bab 4.2

Analisis eksploratif pada penelitian ini dilakukan menggunakan dua dataset aktual yang berbeda, yaitu UCI Mushroom Dataset pada `data/raw/uci/mushroom.csv` dan Secondary Mushroom Dataset pada `data/raw/secondary/secondary_mushroom.csv`. Dengan demikian, secondary dataset dalam proyek ini bukan salinan dari UCI dataset dan harus diperlakukan sebagai sumber data yang berbeda baik dari sisi struktur fitur maupun pola statistiknya.

Pada UCI Mushroom Dataset, inspeksi awal menunjukkan bahwa dataset memiliki 8124 baris dan 23 kolom, dengan missing value hanya pada fitur `stalk-root`. Sesuai pipeline preprocessing proyek, fitur `stalk-root` tetap dipertahankan dan missing value pada fitur kategorikal diimputasi sebagai `unknown`. Analisis asosiasi kategorikal terhadap target pada train split menunjukkan bahwa `odor` memiliki asosiasi paling kuat terhadap kelas dengan nilai Cramer's V sebesar 0.970, diikuti `spore-print-color` sebesar 0.755 dan `gill-color` sebesar 0.679. Temuan ini menunjukkan bahwa ketiga fitur tersebut sangat relevan untuk analisis lanjutan, tetapi temuan tersebut tidak boleh langsung diterjemahkan sebagai struktur root node model Decision Tree karena model belum dilatih pada tahap EDA ini.

Pada Secondary Mushroom Dataset, analisis dilakukan pada data aktual berukuran 61069 baris dan 21 kolom sebelum cleaning. Sebanyak 146 exact duplicate rows dihapus. Selanjutnya, kolom `spore-print-color`, `stem-root`, `veil-color`, dan `veil-type` di-drop karena memiliki missing value lebih dari 80 persen berdasarkan train split, konsisten dengan aturan cleaning Milestone 1. Untuk fitur lain, missing value kategorikal diimputasi sebagai `unknown`, sedangkan missing value numerik diimputasi dengan median pada train split.

Statistik numerik Secondary menunjukkan bahwa kelas edible cenderung memiliki nilai `cap-diameter`, `stem-height`, dan terutama `stem-width` yang lebih besar dibanding kelas poisonous pada train split. Hasil ranking menggunakan `mutual_info_classif` memperlihatkan bahwa `stem-width` menjadi fitur paling informatif dengan skor 0.0639, diikuti `stem-height` sebesar 0.0444, `stem-surface` sebesar 0.0421, `stem-color` sebesar 0.0406, dan `cap-diameter` sebesar 0.0357. Hasil ini menunjukkan bahwa atribut batang memegang peran penting dalam sinyal klasifikasi pada secondary dataset.

Metodologi EDA ini tidak menggunakan `LabelEncoder` untuk fitur nominal dan tidak menghitung korelasi Spearman atas seluruh fitur kategorikal. Sebagai gantinya, asosiasi kategorikal pada UCI dihitung menggunakan tabel kontingensi, chi-square, dan Cramer's V, sedangkan ranking fitur campuran pada Secondary dihitung menggunakan `mutual_info_classif` dengan pemisahan fitur numerik dan kategorikal yang benar. Pendekatan ini dipilih agar interpretasi statistik tetap sesuai dengan tipe data aktual masing-masing fitur.

## Catatan Interpretasi

- `odor` memang sangat dominan pada analisis bivariatif UCI, tetapi itu belum membuktikan bahwa fitur tersebut akan menjadi root node model pohon keputusan.
- Secondary dataset memiliki pola informasi yang berbeda dari UCI dan tidak boleh disimpulkan dengan narasi yang sama.
- Klaim metrik model seperti recall `99%` harus dipindahkan ke bab evaluasi model dan hanya boleh ditulis jika sudah ada hasil eksperimen yang dapat diverifikasi.
